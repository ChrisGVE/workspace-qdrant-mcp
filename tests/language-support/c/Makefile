CC = cc
CFLAGS = -Wall -Wextra -std=c11
SRCS = src/models.c src/storage.c src/utils.c src/main.c
TARGET = bookshelf

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

.PHONY: clean
