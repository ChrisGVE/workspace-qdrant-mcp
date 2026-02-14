// D Language Test File - Regression test for .d language detection
// Expected: file_type=code, language=d

import std.stdio;
import std.algorithm;
import std.range;

struct Point {
    double x, y;

    double distanceTo(Point other) {
        import std.math : sqrt;
        return sqrt((x - other.x) ^^ 2 + (y - other.y) ^^ 2);
    }
}

auto fibonacci() {
    return recurrence!((a, n) => a[n-1] + a[n-2])(0, 1);
}

void main() {
    auto fib = fibonacci().take(10);
    writeln("First 10 Fibonacci numbers:");
    foreach (n; fib) {
        write(n, " ");
    }
    writeln();

    auto p1 = Point(0.0, 0.0);
    auto p2 = Point(3.0, 4.0);
    writefln("Distance: %.2f", p1.distanceTo(p2));
}
