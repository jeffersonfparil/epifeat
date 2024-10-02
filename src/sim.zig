const std: type = @import("std");
const ArrayList: type = std.ArrayList;

const SimError: type = error{
    InputParameterOutOfBounds,
    OutOfMemory,
};
const NumericData: type = union { some: f64, none: null };

const Array: type = struct {
    n: usize,
    p: usize,
    A: ArrayList(ArrayList(NumericData)),

    fn mult_scalar(self: *Array, a: f64) !void {
        for (self.list) |row| {
            for (row.list) |element| {
                element *= a;
            }
        }
        return;
    }
};

/// d = sqrt(transpose(x-u) * inverse(S) * (x-u))
pub fn mahalanobis_distance(d: f64, u: ArrayList(f64), S: ArrayList(ArrayList(f64))) !ArrayList(f64) {
    const n: usize = d.capacity;
    var buffer: [n]f64 = undefined;
    var fixed_buffer_allocator = std.heap.FixedBufferAllocator(&buffer);
    const allocator = fixed_buffer_allocator.alloc();
    var x: ArrayList(f64) = ArrayList(f64).init(allocator);
    defer x.deinit();
    try x.ensureTotalCapacityPrecise(n);
    try x.append(u[0]);
    try x.append(S[0][0]);
    return x;
}

const SimData: type = struct {
    genotype_matrix: ArrayList(ArrayList(NumericData)),
    phenotype_matrix: ArrayList(ArrayList(NumericData)),
    X: ArrayList(ArrayList(f64)),
    b: ArrayList(f64),
    e: ArrayList(f64),

    // TODO:
    // fn init_genotype(
    //     self: *SimData,
    //     n: u64,
    //     p: u64,
    //     genome_size_bp: u64,
    //     chromosome_sizes_bp: []const u64,
    // ) SimError!void {}

    // TODO:
    // fn init_phenotype(
    //     self: *SimData,
    //     a: u64,
    //     b: u64,
    // ) SimError!void {}

    fn shape(self: *SimData) SimError!struct { n: u64, p: u64 } {
        const n: u64 = @as(u64, self.genotype_matrix.capacity());
        if (n < 1) {
            return SimError.InputParameterOutOfBounds;
        }
        const p: u64 = @as(u64, self.genotype_matrix[0].capacity());
        if (p < 1) {
            return SimError.InputParameterOutOfBounds;
        }
        return .{ .n = n, .p = p };
    }
};

test "sim" {}
