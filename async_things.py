import asyncio

async def some_api_call(payload:int):
    """
    This function simulates an API call that takes 2 seconds to complete.
    :param payload:
    :return:
    """
    print(f"start {payload}")
    await asyncio.sleep(2)
    print(f"end {payload}")


tasks = []
for i in range(3):
    tasks.append(some_api_call(i))

async def main():
    await asyncio.gather(*tasks)

asyncio.run(main())
# print(main())

with open("some.txt", 'r') as f:
    print(f.read())

