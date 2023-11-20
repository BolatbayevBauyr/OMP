#include <unordered_map>
#include <omp.h>
#include "helpers.hpp"

unsigned long SequenceInfo::gpsa_sequential(float** S, float** SUB, std::unordered_map<char, int>& cmap) {
    unsigned long visited = 0;
    gap_penalty = SUB[0][cmap['*']]; // min score

	// Boundary
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
		visited++;
	}

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
		visited++;
	}

	// Main part
	for (unsigned int i = 1; i < rows; i++) {
		for (unsigned int j = 1; j < cols; j++) {
			float match = S[i - 1][j - 1] + SUB[ cmap.at(X[i - 1]) ][ cmap.at(Y[j-1]) ];
			float del = S[i - 1][j] + gap_penalty;
			float insert = S[i][j - 1] + gap_penalty;
			S[i][j] = std::max({match, del, insert});

			visited++;
		}
	}

    return visited;
}

unsigned long SequenceInfo::gpsa_taskloop(float** S, float** SUB, std::unordered_map<char, int> cmap, int grain_size) {
    unsigned long visited = 0;
    gap_penalty = SUB[0][cmap['*']];

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Boundary Initialization
            #pragma omp taskloop grainsize(grain_size)
            for (unsigned int i = 1; i < rows; i++) {
                S[i][0] = i * gap_penalty;
                #pragma omp atomic
                visited++;
            }

            #pragma omp taskloop grainsize(grain_size)
            for (unsigned int j = 1; j < cols; j++) {
                S[0][j] = j * gap_penalty;
                #pragma omp atomic
                visited++;
            }

            // Main loop
            #pragma omp taskloop collapse(2) grainsize(grain_size)
            for (unsigned int i = 1; i < rows; i++) {
                for (unsigned int j = 1; j < cols; j++) {
                    float match = S[i - 1][j - 1] + SUB[cmap.at(X[i - 1])][cmap.at(Y[j-1])];
                    float del = S[i - 1][j] + gap_penalty;
                    float insert = S[i][j - 1] + gap_penalty;
                    S[i][j] = std::max({match, del, insert});
                    #pragma omp atomic
                    visited++;
                }
            }
        }
    }
    return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float** S, float** SUB, std::unordered_map<char, int> cmap, int grain_size) {
    unsigned long visited = 0;
    gap_penalty = SUB[0][cmap['*']];

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Boundary Initialization
            for (unsigned int i = 1; i < rows; i += grain_size) {
                #pragma omp task
                {
                    for (unsigned int k = i; k < std::min(i + grain_size, static_cast<unsigned int>(rows)); ++k) {
                        S[k][0] = k * gap_penalty;
                        #pragma omp atomic
                        visited++;
                    }
                }
            }

            for (unsigned int j = 1; j < cols; j += grain_size) {
                #pragma omp task
                {
                    for (unsigned int k = j; k < std::min(j + grain_size, static_cast<unsigned int>(cols)); ++k) {
                        S[0][k] = k * gap_penalty;
                        #pragma omp atomic
                        visited++;
                    }
                }
            }

            // Main loop
            for (unsigned int i = 1; i < rows; i += grain_size) {
                for (unsigned int j = 1; j < cols; j += grain_size) {
                    #pragma omp task
                    {
                        for (unsigned int ii = i; ii < std::min(i + grain_size, static_cast<unsigned int>(rows)); ++ii) {
                            for (unsigned int jj = j; jj < std::min(j + grain_size, static_cast<unsigned int>(cols)); ++jj) {
                                float match = S[ii - 1][jj - 1] + SUB[cmap.at(X[ii - 1])][cmap.at(Y[jj-1])];
                                float del = S[ii - 1][jj] + gap_penalty;
                                float insert = S[ii][jj - 1] + gap_penalty;
                                S[ii][jj] = std::max({match, del, insert});
                                #pragma omp atomic
                                visited++;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp taskwait
    }
    return visited;
}
