//! qlib-rs — Alpha158 因子的 Python 绑定（PyO3）
//!
//! 从 Python 以 `import qlib_rs` 调用 Rust 因子引擎。
//!
//! 设计要点：
//! - 沿用现有 `alpha158-core` 的 pipeline，不重复实现因子逻辑
//! - 输入接受本地 Parquet 路径（与 CLI 完全一致的 schema）
//! - 输出直接返回 NumPy 二维数组 (n_rows, 158)，避免 Python 侧再解析 Parquet
//! - 长计算释放 GIL，便于 Python 多线程场景并发调用

use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use alpha158_data::schema::FACTOR_NAMES;
use alpha158_data::stock::StockOutput;

/// 把 Vec<StockOutput> 展平成 行(row) x 158 的 f32 矩阵 + 行索引
///
/// 行顺序：按股票分组，组内按日期升序（与 Parquet 输出一致）
fn flatten_outputs(
    outputs: &[StockOutput],
) -> (Vec<f32>, Vec<String>, Vec<i32>) {
    let n_factors = FACTOR_NAMES.len();
    let n_rows: usize = outputs.iter().map(|o| o.dates.len()).sum();

    let mut values: Vec<f32> = Vec::with_capacity(n_rows * n_factors);
    let mut codes: Vec<String> = Vec::with_capacity(n_rows);
    let mut dates: Vec<i32> = Vec::with_capacity(n_rows);

    for out in outputs {
        let n = out.dates.len();
        if out.data.len() != n_factors {
            // 结构性问题：宁可显式失败，也不要静默产出宽度不对的矩阵
            panic!(
                "stock {} has {} factor columns, expected {}",
                out.code,
                out.data.len(),
                n_factors
            );
        }
        for i in 0..n {
            codes.push(out.code.clone());
            dates.push(out.dates[i]);
        }
        // 行优先展开
        for i in 0..n {
            for col in out.data.iter() {
                values.push(col[i]);
            }
        }
    }
    (values, codes, dates)
}

/// 计算 Alpha158 因子（从 Parquet 读，返回 NumPy 数组）
///
/// 参数
/// ----
/// input : str
///     输入 Parquet 路径，需含 date/code/open/high/low/close/volume/vwap
/// stock_day : int, 默认 0
///     只输出最近 N 个交易日（0 = 全部）。实际读取 N+60 天以覆盖滚动窗口。
/// threads : int, 默认 0
///     0 = Rayon 自动检测
///
/// 返回
/// ----
/// (values, codes, dates, factor_names)
///     values : np.ndarray, shape=(n_rows, 158), dtype=float32
///     codes  : list[str], 长度 n_rows
///     dates  : np.ndarray, shape=(n_rows,), dtype=int32 (YYYYMMDD)
///     factor_names : list[str], 长度 158
#[pyfunction]
#[pyo3(signature = (input, stock_day = 0, threads = 0))]
fn compute_factors<'py>(
    py: Python<'py>,
    input: &str,
    stock_day: usize,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Vec<String>,
    Bound<'py, numpy::PyArray1<i32>>,
    Vec<String>,
)> {
    let input_path = input.to_string();

    // 长计算期间释放 GIL
    let (values, codes, dates) = py.detach(move || -> PyResult<_> {
        // 线程池只能设置一次；重复设置会报错，这里忽略"已设置"的情况
        if threads > 0 {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global();
        }

        let max_days = if stock_day > 0 {
            Some(stock_day + 60)
        } else {
            None
        };

        let stocks = alpha158_data::reader::read_parquet(&input_path, max_days)
            .map_err(|e| PyRuntimeError::new_err(format!("read_parquet failed: {e}")))?;

        let mut outputs = alpha158_core::engine::compute_all(&stocks);

        if stock_day > 0 {
            outputs = outputs
                .into_iter()
                .map(|mut o| {
                    let total = o.dates.len();
                    if total > stock_day {
                        let skip = total - stock_day;
                        o.dates = o.dates[skip..].to_vec();
                        for col in &mut o.data {
                            *col = col[skip..].to_vec();
                        }
                    }
                    o
                })
                .collect();
        }

        Ok(flatten_outputs(&outputs))
    })?;

    let n_factors = FACTOR_NAMES.len();
    let n_rows = dates.len();
    if n_rows == 0 {
        return Err(PyValueError::new_err("no rows computed (empty input?)"));
    }

    // Vec<f32> -> (n_rows, 158)
    let arr = numpy::ndarray::Array2::from_shape_vec((n_rows, n_factors), values)
        .map_err(|e| PyRuntimeError::new_err(format!("reshape failed: {e}")))?;

    let names: Vec<String> = FACTOR_NAMES.iter().map(|s| s.to_string()).collect();

    Ok((
        arr.into_pyarray(py),
        codes,
        dates.into_pyarray(py),
        names,
    ))
}

/// 返回 158 个因子名（顺序与 compute_factors 输出的列一致）
#[pyfunction]
fn factor_names() -> Vec<String> {
    FACTOR_NAMES.iter().map(|s| s.to_string()).collect()
}

/// 引擎版本 / 因子数自检
#[pyfunction]
fn engine_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_factors", FACTOR_NAMES.len())?;
    d.set_item("crate_version", env!("CARGO_PKG_VERSION"))?;
    Ok(d)
}

#[pymodule]
fn qlib_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_factors, m)?)?;
    m.add_function(wrap_pyfunction!(factor_names, m)?)?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    Ok(())
}
