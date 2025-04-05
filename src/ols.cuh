#include "cuda_runtime.h"

#include <vector>

class OLS {
public:
	OLS(const int dimension);
	~OLS();

	void add_sample(const float* sample, const float expected);
	const std::vector<float> retrieve_estimator();

	cudaError_t cudaStatus() const;

	int dimension() const
	{ return m_dimension; }

private:
	void allocate_cuda_memory();
	void clear_cuda_memory();
	void update_status_and_synchronize();

	void compute_output();

	bool m_is_estimator_dirty;
	cudaError_t m_cudaStatus;
	const int m_dimension;

	float* m_sample;
	float* m_rhs;
	float* m_mat_to_invert;

	// Reserved for the computation of the estimator
	float* m_tmp_mat;
	float* m_inverted_mat;
	float* m_output;

	std::vector<float> m_return;
};
