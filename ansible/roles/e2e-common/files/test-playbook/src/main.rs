use std::{thread, time};


fn main() {
    println!("Start!");

    let ten_secs = time::Duration::from_millis(60000);

    thread::sleep(ten_secs);

    println!("Stop!");
}
