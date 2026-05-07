---
layout: post.njk
title: "Go 并发编程入门"
date: 2026-05-03
description: "探索 goroutine、channel 以及 Go 语言优雅的并发原语，了解为何 Go 是并行编程的理想选择。"
tags:
  - post
  - go
  - concurrency
  - programming
---

Go 的并发方法是其最独特和最受欢迎的特性之一。Go 引入了 *goroutine* 和 *channel*，无需依赖传统的线程模型，就能让编写并发代码变得轻而易举。<sup class="footnote-ref"><a href="#fn1">[1]</a></sup>

## Goroutine：轻量级并发

创建 goroutine 非常简单，只需在函数调用前加上 `go` 关键字：<sup class="footnote-ref"><a href="#fn2">[2]</a></sup>

```go
func main() {
    go sayHello()
    go fetchData("https://api.example.com/data")
    time.Sleep(time.Second)
}
```

<aside id="fn1" class="footnote">
  <p>"Goroutine" 一词由 Rob Pike 创造，他将其描述为"由 Go 运行时管理的轻量级线程"。</p>
</aside>

<aside id="fn2" class="footnote">
  <p>创建 goroutine 就像在函数调用前加 `go` 一样简单——无需任何模板代码。</p>
</aside>

与 OS 线程不同，goroutine 被多路复用到较少的 OS 线程上。Go 运行时处理调度，允许数千甚至数百万个 goroutine 并发运行而不会出现问题。<sup class="footnote-ref"><a href="#fn3">[3]</a></sup>

![Go 并发可视化](./img/1.png)

## Channel：Goroutine 之间的通信

如果 goroutine 是参与者，那么 channel 就是连接它们的消息总线。Channel 提供类型安全的通信和同步：<sup class="footnote-ref"><a href="#fn4">[4]</a></sup>

```go
ch := make(chan string)

go func() {
    result := doWork()
    ch <- result  // 发送到 channel
}()

response := <-ch  // 从 channel 接收
```

<aside id="fn3" class="footnote">
  <p>实际上，一个 Go 进程可以轻松处理 10,000+ 个并发的 goroutine，且开销极小。</p>
</aside>

<aside id="fn4" class="footnote">
  <p>无缓冲 channel 在发送和接收时都会阻塞，提供隐式同步。缓冲 channel 解耦了发送者和接收者。</p>
</aside>

这种模式消除了一整类 bug。对于基本的生产者-消费者场景，你不再需要互斥锁和条件变量——channel 为你处理同步。

## Select 语句

当你有多个 channel 时，`select` 可以让你同时等待所有 channel：<sup class="footnote-ref"><a href="#fn5">[5]</a></sup>

```go
select {
case msg1 := <-ch1:
    fmt.Println("从 channel 1 收到消息:", msg1)
case msg2 := <-ch2:
    fmt.Println("从 channel 2 收到消息:", msg2)
case <-time.After(time.Second):
    fmt.Println("超时")
}
```

<aside id="fn5" class="footnote">
  <p>`select` 语句对于 channel 的作用，就像 `switch` 对于条件语句的作用——但当多个 case 同时就绪时，选择是不确定的。</p>
</aside>

## 实战示例：并发爬虫

这是一个真实的例子——并发网页爬虫，同时获取多个页面：

```go
func fetchAll(urls []string) []string {
    results := make(chan string, len(urls))
    var wg sync.WaitGroup

    for _, url := range urls {
        wg.Add(1)
        go func(u string) {
            defer wg.Done()
            resp, _ := http.Get(u)
            body, _ := io.ReadAll(resp.Body)
            results <- string(body)
        }(url)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    var output []string
    for r := range results {
        output = append(output, r)
    }
    return output
}
```

<aside id="fn6" class="footnote">
  <p>此模式使用 WaitGroup 跟踪完成，并使用单独的 goroutine 在所有获取操作完成后关闭 channel。你也可以添加超时 channel 以防止无限等待。</p>
  <p><img src="https://picsum.photos/seed/go-pattern/300/150" alt="模式图"></p>
</aside>

## 结论

Go 的并发模型成功源于其理念：**通过共享内存来通信**。与其担心锁和互斥锁，不如通过 channel 传递数据。

这种方法从简单脚本到生产系统都适用。Google、Dropbox 和 Twitch 等公司使用 Go，正是因为其并发模型使复杂的并行系统变得易于理解。

下次你需要处理多个并发操作时，考虑使用 goroutine 和 channel。未来的你会感谢现在的你。
