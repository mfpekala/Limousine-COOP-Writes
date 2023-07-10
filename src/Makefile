compile:
	cd build; cmake ..; make && cd ..

runner:
	cd build; cmake ..; make && ./coop-pgm

valgrind-full:
	valgrind --leak-check=full \
					 --show-leak-kinds=all \
					 --track-origins=yes \
					 --verbose \
					 --log-file=valgrind-out.txt \
					./build/Playground