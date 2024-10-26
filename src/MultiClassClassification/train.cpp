#include <iostream>
#include <cstdio>
#include <iterator>
#include <limits>
#include <iomanip>
#include <chrono>
#include <signal.h>
#include <filesystem>
#include <torch/torch.h>
#include "hyperparameters.hpp"
#include "model.hpp"
#include "../raw_data_writer/raw_data_writer.hpp"

int main() {

    RawDataWriter writer;

    // ================== TRAINED MODEL FILE NAMES ==================

    std::string BEST_VALIDATION_MODEL = "best_v_model.pt";
    std::string BEST_TESTING_MODEL    = "best_t_model.pt";

    // ================== TRAINING DATASET ==================

    auto raw_training_dataset = torch::data::datasets::MNIST("data/MNIST/raw", torch::data::datasets::MNIST::Mode::kTrain);
    
    auto training_dataset = raw_training_dataset.map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());

    // RandomSampler (shuffle = true) vs SequentialSampler(shuffle = false)
    auto training_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(training_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(hyperparams::mini_batch_size)
            .workers(2)
    );

    // ================== VALIDATION DATASET ==================

#if defined(VALIDATION_ENABLED)

    auto raw_validation_dataset = torch::data::datasets::MNIST("data/MNIST/raw", torch::data::datasets::MNIST::Mode::kTest);
    
    auto validation_dataset = raw_validation_dataset.map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());

    auto validation_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(validation_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(hyperparams::mini_batch_size)
            .workers(2)
    );

#endif

    // ================== TEST DATASET ==================

#if defined(TESTING_ENABLED)

    auto raw_test_dataset = torch::data::datasets::MNIST("data/MNIST/raw", torch::data::datasets::MNIST::Mode::kTest);
    
    auto test_dataset = raw_test_dataset.map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());

    auto test_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(hyperparams::mini_batch_size)
            .workers(2)
    );

#endif

    // ================== INITIALIZE MODEL, OPTIMIZER, LOSS FUNCTION ==================

    auto model = std::make_shared<MultiClassClassification>();
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(hyperparams::learning_rate));
    torch::nn::CrossEntropyLoss loss_function;

    // ================== GET BEST AVAILABLE COMPUTING DEVICE ==================

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    } else if (torch::mps::is_available()) {
        device = torch::Device(torch::kMPS);
    }

    model->to(device);

    std::cout << "Computing Device: " << device << '\n';

    // ================== LOAD CHECKPOINT IF EXIST ==================

    if (std::filesystem::exists(BEST_TESTING_MODEL) || std::filesystem::exists(BEST_VALIDATION_MODEL)) {
        
        std::cout << "Checkpoint found, loading model...\n";
        
        torch::serialize::InputArchive archive;

        if (std::filesystem::exists(BEST_TESTING_MODEL)) {
            archive.load_from(BEST_TESTING_MODEL);
        } else {
            archive.load_from(BEST_VALIDATION_MODEL);
        }

        model->load(archive);
        optimizer.load(archive);

        std::cout << "Checkpoint loaded successfully\n";
    } else {
        std::cout << "No checkpoint found, starting training from scratch.\n";
    }

    std::cout << '\n';

    // ================== DISPLAY SOME DATASET INFO ==================

    int total_training_batches = std::distance(training_dataloader->begin(), training_dataloader->end());
    std::cout << "training_dataset.size() = " << raw_training_dataset.size().value_or(0) <<'\n';
    std::cout << "training_loader batches = " << total_training_batches <<"\n\n";

#if defined(VALIDATION_ENABLED)

    int total_validation_batches = std::distance(validation_dataloader->begin(), validation_dataloader->end());
    std::cout << "validation_dataset.size() = " << raw_validation_dataset.size().value_or(0) <<'\n';
    std::cout << "validation_loader batches = " << total_validation_batches <<"\n\n";

#endif

#if defined(TESTING_ENABLED)

    int total_test_batches = std::distance(test_dataloader->begin(), test_dataloader->end());
    std::cout << "test_dataset.size() = " << raw_test_dataset.size().value_or(0) <<'\n';
    std::cout << "test_loader batches = " << total_test_batches <<"\n\n";

#endif

    // ================== START OF TRAINING EPOCHS ==================

    auto start_time = std::chrono::high_resolution_clock::now();

    float validation_best_loss = std::numeric_limits<float>::max();
    float test_best_loss       = std::numeric_limits<float>::max();

    for (int epoch = 0; epoch < hyperparams::training_epochs; epoch++) {

        // ================== TRAINING THE MODEL ==================

        float training_amassed_loss = 0.f;
        float training_correct_pred = 0.f;

        int training_mini_batch_idx = 0;
        int training_overall_samples_done = 0;
        int training_log_samples_done = 0;
        int training_log_counts = 0;

        for (torch::data::Example<>& training_mini_batch : *training_dataloader) {
            
            model->train();

            training_mini_batch.data = training_mini_batch.data.to(device);
            training_mini_batch.target = training_mini_batch.target.to(device);

            optimizer.zero_grad();

            // batch.data | Shape: [Batch x Channel x 28 x 28]
            auto training_mini_batch_output = model->forward(training_mini_batch.data);

            // output       | Shape: [Batch, 10]
            // batch.target | Shape: [Batch]
            // BinaryCrossEntropyLoss
            auto training_mini_batch_loss = loss_function(training_mini_batch_output, training_mini_batch.target);
            training_mini_batch_loss.backward();
            optimizer.step();

#if defined(VALIDATION_ENABLED)

            // accumulate training data every mini batches

            training_amassed_loss += training_mini_batch_loss.item<float>();
            training_correct_pred += training_mini_batch_output.argmax(1).eq(training_mini_batch.target).sum().item<int>();
            training_overall_samples_done += training_mini_batch.data.size(0);
            training_log_samples_done += training_mini_batch.data.size(0);
            training_log_counts++;

            // validate the network after a certain mini batch number

            if (!((training_mini_batch_idx + 1) % hyperparams::mini_batch_log_interval) || training_mini_batch_idx == 0) {

                // calculate running averages of training data

                auto training_running_loss = training_amassed_loss / static_cast<float>(training_log_counts);
                auto training_running_accuracy = training_correct_pred / training_log_samples_done;

                writer.add_scalar("Training Loss", training_running_loss, epoch * total_training_batches + training_mini_batch_idx);
                writer.add_scalar("Training Accuracy", training_running_accuracy, epoch * total_training_batches + training_mini_batch_idx);

                std::printf(
                    "Training: Batch [%d/%d] | Epoch : %d/%d | Accuracy: %.2f%% | Loss: %.7f | Dataset: [%5d/%5d]\n",
                    training_mini_batch_idx + 1,
                    total_training_batches,
                    epoch + 1,
                    hyperparams::training_epochs,
                    training_running_accuracy * 100,
                    training_running_loss,
                    training_overall_samples_done,
                    static_cast<int>(raw_training_dataset.size().value_or(0))
                );

                // ================== VALIDATING THE MODEL ==================

                float validation_running_loss = 0.f;
                float validation_correct_pred = 0.f;
                float validation_samples_done = 0.f;

                int validation_mini_batch_idx = 0;

                for (torch::data::Example<>& validation_batch : *validation_dataloader) {
                    
                    model->eval();
                    
                    validation_batch.data = validation_batch.data.to(device);
                    validation_batch.target = validation_batch.target.to(device);

                    auto validation_batch_output = model->forward(validation_batch.data);
                    auto validation_batch_loss = loss_function(validation_batch_output, validation_batch.target);
                
                    validation_running_loss += validation_batch_loss.item<float>();
                    validation_correct_pred += validation_batch_output.argmax(1).eq(validation_batch.target).sum().item<int>();
                    validation_samples_done += validation_batch.data.size(0);
                
                    validation_mini_batch_idx++;
                }

                auto validation_ave_loss = validation_running_loss / static_cast<float>(validation_mini_batch_idx);
                auto validation_accuracy = validation_correct_pred / validation_samples_done;
            
                writer.add_scalar("Validation Loss", validation_ave_loss, epoch * total_training_batches + training_mini_batch_idx);
                writer.add_scalar("Validation Accuracy", validation_accuracy, epoch * total_training_batches + training_mini_batch_idx);

                std::printf(
                    "Validation: Batch [%d/%d] | Epoch : %d/%d | Accuracy: %.2f%% | Loss: %.7f | Dataset: [%5d/%5d]\n\n",
                    validation_mini_batch_idx,
                    total_validation_batches,
                    epoch + 1,
                    hyperparams::training_epochs,
                    validation_accuracy * 100,
                    validation_ave_loss,
                    static_cast<int>(validation_samples_done),
                    static_cast<int>(raw_validation_dataset.size().value_or(0))
                );

                if (validation_ave_loss < validation_best_loss) {
                    validation_best_loss = validation_ave_loss;

                    torch::serialize::OutputArchive archive;
                    model->save(archive);
                    optimizer.save(archive);
                    archive.save_to(BEST_VALIDATION_MODEL);

                    std::cout << "new best VALIDATION model saved!\n\n";
                }

                // compare training and validation losses
                writer.add_scalars("Training vs Validation Loss", {
                    {"Training",   training_running_loss},
                    {"Validation", validation_ave_loss}
                }, epoch * total_training_batches + training_mini_batch_idx);

                // compare training and validation accuracy
                writer.add_scalars("Training vs Validation Accuracy", {
                    {"Training",   training_running_accuracy},
                    {"Validation", validation_accuracy}
                }, epoch * total_training_batches + training_mini_batch_idx);

                // zero out accumulated training data to get the next proper running averages

                training_amassed_loss = 0.f;
                training_correct_pred = 0.f;

                training_log_counts = 0;
                training_log_samples_done = 0.f;

                // ================== VALIDATION : END ==================
            }
#endif

            training_mini_batch_idx++;
        }

        // ================== TRAINING : END ==================

        // ================== TESTING THE MODEL ==================

#if defined(TESTING_ENABLED)

        float test_amassed_loss = 0.f;
        float test_correct_pred = 0.f;
        float test_samples_done = 0.f;

        int test_mini_batch_idx = 0;

        for (torch::data::Example<>& test_batch : *test_dataloader) {
            
            model->eval();
            
            test_batch.data = test_batch.data.to(device);
            test_batch.target = test_batch.target.to(device);

            auto test_batch_output = model->forward(test_batch.data);
            auto test_batch_loss = loss_function(test_batch_output, test_batch.target);
        
            test_amassed_loss += test_batch_loss.item<float>();
            test_correct_pred += test_batch_output.argmax(1).eq(test_batch.target).sum().item<int>();
            test_samples_done += test_batch.data.size(0);

            test_mini_batch_idx++;
        }

        auto test_ave_loss = test_amassed_loss / static_cast<float>(test_mini_batch_idx);
        auto test_accuracy = test_correct_pred / test_samples_done;
    
        writer.addScalar("Testing Loss", test_ave_loss, epoch);
        writer.addScalar("Testing Accuracy", test_accuracy, epoch);

        std::printf(
            "Testing: Batch [%d/%d] | Epoch : %d/%d | Accuracy: %.2f%% | Loss: %.7f | Dataset: [%5d/%5d]\n\n",
            test_mini_batch_idx,
            total_test_batches,
            epoch + 1,
            hyperparams::training_epochs,
            test_accuracy * 100,
            test_ave_loss,
            static_cast<int>(test_samples_done),
            static_cast<int>(raw_test_dataset.size().value_or(0))
        );

        if (test_ave_loss < test_best_loss) {
            test_best_loss = test_ave_loss;

            torch::serialize::OutputArchive archive;
            model->save(archive);
            optimizer.save(archive);
            archive.save_to(BEST_TESTING_MODEL);

            std::cout << "new best TESTING model saved!\n\n";
        }
        
#endif

        // ================== TESTING : END ==================

        std::cout << '\n';
    }

    // ================== END OF TRAINING EPOCHS ==================

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    size_t seconds = total_time.count();
    float minutes = static_cast<float>(total_time.count()) / 60.f;
    float hours = minutes / 60.f;

    std::cout << "total runtime time : " << seconds << " sec(s), " << minutes << " min(s), " << hours << " hr(s)\n";

    return 0;
}
