const assert = require('assert');
const sd = require('../src/js/index');

console.log('=== node-stable-diffusion-cpp basic tests ===\n');

// Test version()
const ver = sd.version();
console.log('version():', ver);
assert.strictEqual(typeof ver, 'string');
assert.ok(ver.length > 0, 'version should not be empty');

// Test commit()
const commit = sd.commit();
console.log('commit():', commit);
assert.strictEqual(typeof commit, 'string');

// Test getSystemInfo()
const sysInfo = sd.getSystemInfo();
console.log('getSystemInfo():', sysInfo);
assert.strictEqual(typeof sysInfo, 'string');
assert.ok(sysInfo.length > 0, 'system info should not be empty');

// Test getNumPhysicalCores()
const cores = sd.getNumPhysicalCores();
console.log('getNumPhysicalCores():', cores);
assert.strictEqual(typeof cores, 'number');
assert.ok(cores > 0, 'should have at least 1 physical core');

// Test that classes exist
assert.strictEqual(typeof sd.StableDiffusionContext, 'function');
assert.strictEqual(typeof sd.UpscalerContext, 'function');
assert.strictEqual(typeof sd.StableDiffusionContext.create, 'function');
assert.strictEqual(typeof sd.UpscalerContext.create, 'function');

// Test that free functions exist
assert.strictEqual(typeof sd.convert, 'function');
assert.strictEqual(typeof sd.preprocessCanny, 'function');
assert.strictEqual(typeof sd.setLogCallback, 'function');
assert.strictEqual(typeof sd.setProgressCallback, 'function');
assert.strictEqual(typeof sd.setPreviewCallback, 'function');

// Test setLogCallback (set and clear)
sd.setLogCallback((data) => {
    // Just verify it doesn't crash
});
sd.setLogCallback(null);

console.log('\n=== All basic tests passed! ===');
