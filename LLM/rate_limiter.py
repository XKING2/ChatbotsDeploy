
import time


class RateLimiter:

    def __init__(self, max_requests=5, period=60):

        self.max_requests = max_requests
        self.period = period
        self.requests = []

    def wait(self):

        now = time.time()

        # hapus request yang sudah lewat 60 detik
        self.requests = [

            t

            for t in self.requests

            if now - t < self.period

        ]

        if len(self.requests) >= self.max_requests:

            sleep_time = self.period - (now - self.requests[0]) + 1

            if sleep_time > 0:

                print(
                    f"\n[RateLimiter] "
                    f"Limit tercapai."
                    f" Menunggu {sleep_time:.1f} detik...\n"
                )

                time.sleep(sleep_time)

            now = time.time()

            self.requests = [

                t

                for t in self.requests

                if now - t < self.period

            ]

        self.requests.append(time.time())