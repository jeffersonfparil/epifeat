const std: type = @import("std");
const Allocator: type = std.mem.Allocator;

fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();
        items: [][]T,
        n: usize,
        p: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, n: usize, p: usize) Self {
            return Self{
                .items = &[n][p]T{},
                .n = n,
                .p = p,
                .allocator = allocator,
            };
        }

        /// Initialize with capacity to hold `num` elements.
        /// The resulting capacity will equal `num` exactly.
        /// Deinitialize with `deinit` or use `toOwnedSlice`.
        pub fn initCapacity(allocator: Allocator, num: usize) Allocator.Error!Self {
            var self = Self.init(allocator);
            try self.ensureTotalCapacityPrecise(num);
            return self;
        }

        /// Release all allocated memory.
        pub fn deinit(self: Self) void {
            if (@sizeOf(T) > 0) {
                self.allocator.free(self.allocatedSlice());
            }
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

    const a = try Matrix(f64).init(allocator, 10, 20);
    std.debug.print("a.items[0]={?}\n", .{a.items[0]});
    // std.debug.print("a.p={?}\n", .{a.p});
    // std.debug.print("a.A.items[0].items[0]={?}\n", .{a.A.items[0].items[0]});
}
