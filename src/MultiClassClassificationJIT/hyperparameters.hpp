#ifndef MCC_MODEL_HYPER_PARAMETERS_HEADER_GUARD
#define MCC_MODEL_HYPER_PARAMETERS_HEADER_GUARD

namespace hyperparams {
    constexpr float learning_rate = 0.001f;
    constexpr int training_epochs = 2;
    constexpr int mini_batch_size = 500;
    constexpr int mini_batch_log_interval = 10;
};

#define VALIDATION_ENABLED 1
// #define TESTING_ENABLED 1

#endif