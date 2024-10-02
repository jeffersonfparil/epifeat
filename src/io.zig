const std = @import("std");

pub fn slices(n: u64) !f64 {
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    const allocator = std.heap.page_allocator;
    var x = std.ArrayList(f64).init(allocator);
    defer x.deinit();
    var i: u64 = 0;
    while (i < n) : (i += 1) {
        const x_new: f64 = rand.float(f64);
        try x.append(x_new);
        std.debug.print("x_new[{?}]={?}\n", .{ i, x_new });
    }
    for (x.items, 0..) |s, j| {
        std.debug.print("x[{?}]={?}\n", .{ j, s });
    }
    const out: f64 = 0.0;
    return out;
}

test "slices" {
    const n: u64 = 10;
    _ = try slices(n);
}
