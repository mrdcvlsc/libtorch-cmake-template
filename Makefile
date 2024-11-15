debug:
	cmake -S src -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=FALSE -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --config Debug -j2

release:
	cmake -S src -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=FALSE -DCMAKE_BUILD_TYPE=Release
	cmake --build build --config Release -j2

serve_tensorboard:
	python3 generate_tensorboard_data.py
	tensorboard --logdir=runs

clean:
	rm -r *.pt raw_data_runs runs

download_datasets:
	curl https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip -o data/cats-vs-dogs.zip -L

extract_datasets:
	unzip data/cats-vs-dogs.zip -d data/