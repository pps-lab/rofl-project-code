use std::env;
use std::fs::File;
use std::fs::OpenOptions;
use std::path::PathBuf;

pub fn get_bench_dir() -> PathBuf {
    let mut cwd = env::current_exe().unwrap();
    cwd.pop();
    cwd.pop();
    cwd.pop();
    cwd.push("benchmarks");
    cwd
}

pub fn create_bench_file(label: &String) -> File {
    let mut bench_file = get_bench_dir();
    let _res = std::fs::create_dir_all(&bench_file);
    bench_file.push(label);
    bench_file.set_extension("bench");
    println!("bench file: {}", bench_file.display());
    let file = match OpenOptions::new()
        .append(true)
        .create(true)
        .open(bench_file)
    {
        Err(err) => panic!("Could not find {}", err),
        Ok(f) => f,
    };
    return file;
}
