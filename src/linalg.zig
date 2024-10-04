const std: type = @import("std");
const Allocator: type = std.mem.Allocator;

fn Matrix(comptime T: type) type {
    return struct {
        data: [][]T = undefined,
        n: usize,
        p: usize,

        const Self = @This();

        pub fn init(n: usize, p: usize, allocator: Allocator) !Self {
            const data = try allocator.alloc([]T, n);
            for (0..n) |i| {
                data[i] = try allocator.alloc(T, p);
            }
            return .{
                .data = data,
                .n = n,
                .p = p,
            };
        }

        pub fn dim(self: Self) [2]usize {
            const x: [2]usize = [2]usize{ self.n, self.p };
            return x;
        }
    };
}

const Array: type = struct {};

test "linalg" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // var x = std.ArrayList(std.ArrayList(?f64)).init(allocator);
    // defer x.deinit();

    // var x_sub = std.ArrayList(?f64).init(allocator);
    // defer x_sub.deinit();

    // try x_sub.append(1.00);
    // try x_sub.append(null);
    // try x.append(x_sub);

    // std.debug.print("x_sub.items[0]={?}\n", .{x_sub.items[0]});
    // std.debug.print("x_sub.items[1]={?}\n", .{x_sub.items[1]});
    // std.debug.print("x.items[0].items[0]={?}\n", .{x.items[0].items[0]});
    // std.debug.print("x.items[0].items[1]={?}\n", .{x.items[0].items[1]});
    // std.debug.print("x.items[0].items[2]={?}\n", .{x.items[0].items[2]});

    const a = try Matrix(f64).init(2, 3, allocator);
    std.debug.print("a.data[0][0]={?}\n", .{a.data[0][0]});
    std.debug.print("a={any}\n", .{a});
    std.debug.print("a.dim()={any}\n", .{a.dim()});
    std.debug.print("a.dim()[0]={any}\n", .{a.dim()[0]});
    std.debug.print("a.dim()[1]={any}\n", .{a.dim()[1]});
    // std.debug.print("a.p={?}\n", .{a.p});
    // std.debug.print("a.A.items[0].items[0]={?}\n", .{a.A.items[0].items[0]});
}
