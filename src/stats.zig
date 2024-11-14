const std: type = @import("std");
const Allocator: type = std.mem.Allocator;
const linalg: type = @import("linalg.zig");
const expect = std.testing.expect;

pub fn random_sampling(n: usize, allocator: Allocator) ![]f64 {
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    var out: []f64 = try allocator.alloc(f64, n);
    for (0..n) |i| {
        out[i] = rand.float(f64);
    }
    return out;
}

/// Numerical approximation of the error function of the standard normal distribution by Yaya Dia in 2023
/// Citation: Dia, Yaya D., Approximate Incomplete Integrals, Application to Complementary Error Function (June 21, 2023). Available at SSRN: https://ssrn.com/abstract=4487559 or http://dx.doi.org/10.2139/ssrn.4487559
pub fn cdf_gaussian(y: f64, lower_tail: bool) f64 {
    var x: f64 = y;
    if (lower_tail) {
        x = y;
    } else {
        x = -y;
    }
    const phi: f64 = 1.00 - (0.39894228040143268 / (x + 2.92678600515804815) *
        (std.math.pow(f64, x, 2.00) + (8.42742300458043240 * x) + 18.38871225773938487) / (std.math.pow(f64, x, 2.00) + (5.81582518933527391 * x) + 8.97280659046817350) *
        (std.math.pow(f64, x, 2.00) + (7.30756258553673541 * x) + 18.25323235347346525) / (std.math.pow(f64, x, 2.00) + (5.70347935898051437 * x) + 10.27157061171363079) *
        (std.math.pow(f64, x, 2.00) + (5.66479518878470765 * x) + 18.61193318971775795) / (std.math.pow(f64, x, 2.00) + (5.51862483025707963 * x) + 12.72323261907760928) *
        (std.math.pow(f64, x, 2.00) + (4.91396098895240075 * x) + 24.14804072812762821) / (std.math.pow(f64, x, 2.00) + (5.26184239579604207 * x) + 16.88639562007936908) *
        (std.math.pow(f64, x, 2.00) + (3.83362947800146179 * x) + 11.61511226260603247) / (std.math.pow(f64, x, 2.00) + (4.92081346632882033 * x) + 24.12333774572479110) *
        @exp(-(std.math.pow(f64, x, 2.00) / 2)));
    return phi;
}

// pub fn sample_univariate_gaussian(mu: f64, sd: f64, size: usize, allocator: Allocator) ![]f64 {}

test "stats" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const x = try random_sampling(10, allocator);
    std.debug.print("test={any}\n", .{x});

    const x0 = cdf_gaussian(0.0, true);
    const x1 = cdf_gaussian(0.5, true);
    const x2 = cdf_gaussian(1.0, true);
    const x3 = cdf_gaussian(2.0, true);
    try expect(@abs(x0 - 0.5000000) < 0.00001);
    try expect(@abs(x1 - 0.6914625) < 0.00001);
    try expect(@abs(x2 - 0.8413447) < 0.00001);
    try expect(@abs(x3 - 0.9772499) < 0.00001);
}
