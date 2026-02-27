const path = require('path');

let binding;

function loadBinding() {
    if (binding) return binding;

    // Try node-gyp-build (prebuilds)
    try {
        binding = require('node-gyp-build')(path.resolve(__dirname, '..', '..'));
        return binding;
    } catch (e) {
        // Fall through
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
            // Fall through
        }
    }

    throw new Error(
        'Could not load node-stable-diffusion-cpp native module. ' +
        'Run `npm run build` to compile from source.'
    );
}

module.exports = loadBinding;
