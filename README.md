# NCFlow

Anonymous code repository for NCFlow.

Setup validated on Ubuntu 16.04.

Run `download.sh` to fetch the traffic matrices and pre-computed paths used in
our evaluation. (For confidentiality reasons, we only share TMs and paths for
topologies from the Internet Topology Zoo.)

## Dependencies
- Python 3.6 (Anaconda installation recommended)
  - See `environment.yml` for a list of Python library dependencies
- Julia 1.0.5 (to run TEAVAR\*)
  - See `../ext/teavar/dependencies.txt` for a list of Julia library dependencies
- Gurobi 8.1.1 (Requires a Gurobi license)

