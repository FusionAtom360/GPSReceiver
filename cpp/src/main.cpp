#include "gpsrx/receiver.hpp"

int main() {
    return gpsrx::ca_code_length == 1023 ? 0 : 1;
}
