const path = require('path');

const PLATFORMS = {
    'darwin-arm64': '@stable-diffusion-cpp-node-api/darwin-arm64',
    'darwin-x64':   '@stable-diffusion-cpp-node-api/darwin-x64',
    'linux-x64':    '@stable-diffusion-cpp-node-api/linux-x64',
    'win32-x64':    '@stable-diffusion-cpp-node-api/win32-x64',
};

let binding;

function loadBinding() {
    if (binding) return binding;

    const key = `${process.platform}-${process.arch}`;
    const pkg = PLATFORMS[key];

    // 1. Try platform-specific npm package
    if (pkg) {
        try {
            binding = require(pkg);
            return binding;
        } catch (_) {}
    }

    // 2. Fallback to local build (development / unsupported platform)
    try {
        binding = require(path.resolve(__dirname, '..', '..', 'build', 'Release', 'node_stable_diffusion.node'));
        return binding;
    } catch (_) {}

    throw new Error(
        `stable-diffusion-cpp-node-api: no prebuilt binary for ${key}.\n` +
        'Supported platforms: ' + Object.keys(PLATFORMS).join(', ') + '.\n' +
        'To build from source, install from git instead:\n' +
        '  npm install git+https://github.com/sir-ingvarr/stable-diffusion-cpp-node-api'
    );
}

module.exports = loadBinding;
