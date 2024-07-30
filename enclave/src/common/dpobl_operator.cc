#include "enclave/dpobl_operator.h"

// Initialize the static members
std::map<int, std::shared_ptr<DummySamples>> DummySamples::instances_;