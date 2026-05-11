---
layout: post.njk
title: "triosim 模拟器 （一）"
date: 2026-05-10
description: "triosim 模拟器（一）"
tags:
  - post
  - triosim-cpp

---

## 1. 互斥锁 `std::mutex`

特点：**同一时间只允许一个线程访问资源**。

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;
int counter = 0;

void add() {
    // 其他线程仍然可以调用 add 函数，但如果函数里也要锁同一个 smtx，就会在加锁那一行等待
    std::lock_guard<std::mutex> lock(mtx); // 自动加锁，离开作用域自动解锁
    counter++;
}

int main() {
    std::thread t1(add);    // 创建一个线程对象 r1，并让新线程执行函数 add
    std::thread t2(add);

    t1.join();              // 当前线程等待 t1 这个线程执行完
    t2.join();

    std::cout << counter << "\n";
}
```

常用写法：

```cpp
std::lock_guard<std::mutex> lock(mtx);
```

---

## 2. 共享锁 `std::shared_mutex`

特点：

- 多个线程可以**同时读**
- 写线程需要**独占访问**
- 适合“读多写少”

需要 C++17：

```cpp
#include <iostream>
#include <thread>
#include <shared_mutex>
#include <vector>

std::shared_mutex smtx;
int data = 0;

// 读：共享锁，多个线程可同时进入
void read_data() {
    std::shared_lock<std::shared_mutex> lock(smtx);
    std::cout << data << "\n";
}

// 写：独占锁，只能一个线程进入
void write_data() {
    std::unique_lock<std::shared_mutex> lock(smtx);
    data++;
}

int main() {
    std::thread r1(read_data);
    std::thread r2(read_data);
    std::thread w1(write_data);

    r1.join();
    r2.join();
    w1.join();
}
```

---

## 3. 简单对比

| 类型 | 用法 | 效果 |
|---|---|---|
| `std::mutex` | `std::lock_guard<std::mutex>` | 读写都互斥 |
| `std::shared_mutex` | `std::shared_lock` | 多个读可同时进行 |
| `std::shared_mutex` | `std::unique_lock` | 写独占 |

---

