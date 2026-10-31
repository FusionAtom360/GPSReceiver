#include "gpsrx/receiver.hpp"

int main() {
    return gpsrx::ca_code_rate_hz > 0.0 ? 0 : 1;
}
