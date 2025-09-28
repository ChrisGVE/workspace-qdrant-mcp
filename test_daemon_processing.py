def hello_world():
    """A simple test function"""
    return "Hello from daemon test!"

def process_data(data):
    """Process some data"""
    if not data:
        return None

    result = []
    for item in data:
        processed = item.upper().strip()
        result.append(processed)

    return result

class TestClass:
    """A simple test class"""

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    print(hello_world())
    print(process_data(["hello", "world", "test"]))

    test_obj = TestClass("Claude")
    print(test_obj.greet())