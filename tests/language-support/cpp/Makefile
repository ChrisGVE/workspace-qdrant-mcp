CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
SRCS = src/models.cpp src/storage.cpp src/utils.cpp src/main.cpp
TARGET = bookshelf

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

.PHONY: clean
