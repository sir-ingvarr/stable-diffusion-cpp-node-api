const path = require('path');

let binding;

function loadBinding() {
    if (binding) return binding;

    const errors = [];

    // Try node-gyp-build (prebuilds)
    try {
        binding = require('node-gyp-build')(path.resolve(__dirname, '..', '..'));
        return binding;
    } catch (e) {
        errors.push('node-gyp-build: ' + e.message);
    }

    // Try cmake-js build output
    const buildPaths = [
        path.resolve(__dirname, '..', '..', 'build', 'Release', 'node_stable_diffusion.node'),
        path.resolve(__dirname, '..', '..', 'build', 'Debug', 'node_stable_diffusion.node'),
        path.resolve(__dirname, '..', '..', 'build', 'node_stable_diffusion.node'),
    ];

    for (const p of buildPaths) {
        try {
            binding = require(p);
            return binding;
        } catch (e) {
            errors.push(path.basename(path.dirname(p)) + ': ' + e.message);
        }
    }

    throw new Error(
        'Could not load node-stable-diffusion-cpp native module. ' +
        'Run `npm run build` to compile from source.\n' +
        'Attempted:\n  ' + errors.join('\n  ')
    );
}

module.exports = loadBinding;
