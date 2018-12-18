#pragma once
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>




// ============================================================================
/**
 * Basic thread pool implementation: https://github.com/progschj/ThreadPool.
 */
class ThreadPool
{
public:


    /**
     * Constructor. The given number of threads are started idling.
     */
    ThreadPool(std::size_t num_threads);


    /**
     * Destructor. Joins all threads, so it'll block if jobs are still
     * running.
     */
    ~ThreadPool();


    /**
     * Enqueue a job to be run, and return a promise.
     */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;


private:
    // ========================================================================
    std::vector<std::thread> workers;           // need to keep track of threads so we can join them
    std::queue<std::function<void()>> tasks;    // the task queue
    std::mutex queue_mutex;                     // synchronization
    std::condition_variable condition;
    bool stop;
};




// ============================================================================
inline ThreadPool::ThreadPool(std::size_t num_threads)
: stop(false)
{
    for (std::size_t i = 0; i < num_threads; ++i)
    {
        workers.emplace_back([this]
        {
            for(;;)
            {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);

                    condition.wait(lock, [this] { return stop || ! tasks.empty(); });

                    if (stop && tasks.empty())
                    {
                        return;
                    }
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
    }
}

inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();

    for (auto &worker: workers)
    {
        worker.join();
    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if (stop)
        {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task] { (*task)(); });
    }
    condition.notify_one();
    return res;
}
