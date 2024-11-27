const std = @import("std");
const expect = std.testing.expect;

pub fn write(fname: []const u8) ![]const u8 {
    var file = try std.fs.cwd().createFile(fname, .{});
    defer file.close();
    const some_text: *const [40]u8 = "Hello world! Writing a text file in zig.";
    try file.writeAll(some_text);
    return fname;
}

test "io" {
    const fname = "test.tmp";
    const out = try write(fname);
    try expect(std.mem.eql(u8, fname, out));
    // Clean-up
    try std.fs.cwd().deleteFile(fname);
}
