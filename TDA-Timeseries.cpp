/* 
   Pre-requisite: file 'data/retClose_emb.txt' contains a d-dim time series. 
   In practice, this time series is generated as the time delay embedding of a
   1-d time series.
   */
#include <gudhi/Rips_complex.h>
#include <gudhi/distance_functions.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/Points_off_io.h>
#include <gudhi/Persistence_landscape.h>
#include <boost/program_options.hpp>
#include <string>
#include <vector>
#include <limits>  // infinity
// Types definition
using Simplex_tree = Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
using Filtration_value = Simplex_tree::Filtration_value;
using Rips_complex = Gudhi::rips_complex::Rips_complex<Filtration_value>;
using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;
using Point = std::vector<double>;
using Points_off_reader = Gudhi::Points_off_reader<Point>;
using Persistence_landscape = Gudhi::Persistence_representations::Persistence_landscape;
void program_options(int argc, char* argv[], std::string& off_file_points, std::string& filediag,
                     Filtration_value& threshold, int& dim_max, int& p, Filtration_value& min_persistence);
double norm_of_landscape(const std::vector<Point> &point_cloud, 
		std::string& filediag,
		Filtration_value& threshold, int dim_max, int p, 
		Filtration_value& min_persistence);

int main(int argc, char* argv[]) {
  std::string off_file_points;
  std::string filediag;
  Filtration_value threshold;
  int dim_max;
  int p;
  Filtration_value min_persistence;
  const int win_size = 50;	// window size

  // auxiliary vars
  int i;

  program_options(argc, argv, off_file_points, filediag, threshold, dim_max, p, min_persistence);
  Points_off_reader off_reader(off_file_points);
  std::vector<Point> timeseries = off_reader.get_point_cloud();
  
  std::vector<Point> S;	// training set
  // Norm of landscape
  for(i=0; i<timeseries.size()-win_size; i++) {
	  std::clog << "\nProcessing point "<< i << " out of " <<
		  timeseries.size()-win_size-1 << std::endl;

	  std::vector<Point> point_cloud = 
		  std::vector<Point>(timeseries.begin()+i, 
				  timeseries.begin()+i+win_size);
	  /*
	  std::clog << "\nSize of point_cloud : " << point_cloud.size() <<
		  std::endl;;
		  */
	  double norm = norm_of_landscape(point_cloud, filediag, threshold, dim_max, p,
		  min_persistence);
	  //std::clog << "\nL^1 Norm of landscape : " << norm << std::endl;

	  // Features
	  Point p = point_cloud.back();	// last point in the point cloud
	  p.push_back(norm);

	  // Label (0 = negative return, 1 = non-negative return)
	  if(timeseries[i+win_size].back() >= 0) 
		  p.push_back(1.0);
	  else
		  p.push_back(0.0);

	  S.push_back(p);
  }

  // Write training set S to file
  std::ofstream outFile("training.txt");
  for (const auto &p : S) {
	  for (const auto x : p)
		  outFile << x << " ";
	  outFile << "\n";
  }

  return(0);
}

double norm_of_landscape(const std::vector<Point> &point_cloud, 
		std::string& filediag,
		Filtration_value& threshold, int dim_max, int p, 
		Filtration_value& min_persistence) {
  Rips_complex rips_complex_from_file(point_cloud, threshold, Gudhi::Euclidean_distance());
  // Construct the Rips complex in a Simplex Tree
  Simplex_tree simplex_tree;
  rips_complex_from_file.create_complex(simplex_tree, dim_max);
  /*
  std::clog << "\nThe complex contains " << simplex_tree.num_simplices() << " simplices \n";
  std::clog << "   and has dimension " << simplex_tree.dimension() << " \n";
  */

  // Compute the persistence diagram of the complex in degree 1
  Persistent_cohomology pcoh(simplex_tree);
  // initializes the coefficient field for homology
  pcoh.init_coefficients(p);
  pcoh.compute_persistent_cohomology(min_persistence);

  /*
  std::clog << "\nPersistence diagram of the complex, " << "\n";
  std::clog << "   using " << p << " homology (1=conn. comp., 2=loops) \n";
  // Output the diagram in filediag
  if (filediag.empty()) {
    pcoh.output_diagram();
  } else {
    std::ofstream out(filediag);
    pcoh.output_diagram(out);
    out.close();
  }
  */

  // Output intervals in dimension 1
  auto intervals = pcoh.intervals_in_dimension(1);

  /*
  std::clog << "\nThere are " << intervals.size() << " intervals in dimension 1: \n";

  for (auto interval : intervals) 
	  std::cout << interval.first << " " << interval.second << std::endl;
	  */

  // Compute the norm of persistence landscape

  // To avoid compiler errors, convert intervals 
  // from vector of <float, float> pairs to 
  // vector of <double, double> pairs.
  std::vector<std::pair<double, double> > persistence;
  for (auto interval : intervals)
	  persistence.push_back(interval);

  // Create a persistence landscape based on the persistence diagram
  // We use only birth-death pairs (intervals) in dimension 1
  Persistence_landscape l(persistence);

  // Output landscape
  /*
  std::clog << "\nlandscape l: " << l << std::endl;
  */

  // Norm of landscape
  return(l.compute_norm_of_landscape(1.));

  // Create file which is suitable for visualization via gnuplot:
  //l.plot("landscape");
}

void program_options(int argc, char* argv[], 
		std::string& off_file_points, std::string& filediag,
		Filtration_value& threshold, int& dim_max, int& p, 
		Filtration_value& min_persistence) {
  namespace po = boost::program_options;
  po::options_description hidden("Hidden options");
  hidden.add_options()("input-file", po::value<std::string>(&off_file_points),
                       "Name of an OFF file containing a point set.\n");
  po::options_description visible("Allowed options", 100);
  visible.add_options()("help,h", "produce help message")(
      "output-file,o", po::value<std::string>(&filediag)->default_value(std::string()),
      "Name of file in which the persistence diagram is written. Default print in std::clog")(
      "max-edge-length,r",
      po::value<Filtration_value>(&threshold)->default_value(std::numeric_limits<Filtration_value>::infinity()),
      "Maximal length of an edge for the Rips complex construction.")(
      "cpx-dimension,d", po::value<int>(&dim_max)->default_value(1),
      "Maximal dimension of the Rips complex we want to compute.")(
      "field-charac,p", po::value<int>(&p)->default_value(11),
      "Characteristic p of the coefficient field Z/pZ for computing homology.")(
      "min-persistence,m", po::value<Filtration_value>(&min_persistence),
      "Minimal lifetime of homology feature to be recorded. Default is 0. Enter a negative value to see zero length "
      "intervals");
  po::positional_options_description pos;
  pos.add("input-file", 1);
  po::options_description all;
  all.add(visible).add(hidden);
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(all).positional(pos).run(), vm);
  po::notify(vm);
  if (vm.count("help") || !vm.count("input-file")) {
    std::clog << std::endl;
    std::clog << "Compute the persistent homology with coefficient field Z/pZ \n";
    std::clog << "of a Rips complex defined on a set of input points.\n \n";
    std::clog << "The output diagram contains one bar per line, written with the convention: \n";
    std::clog << "   p   dim b d \n";
    std::clog << "where dim is the dimension of the homological feature,\n";
    std::clog << "b and d are respectively the birth and death of the feature and \n";
    std::clog << "p is the characteristic of the field Z/pZ used for homology coefficients." << std::endl << std::endl;
    std::clog << "Usage: " << argv[0] << " [options] input-file" << std::endl << std::endl;
    std::clog << visible << std::endl;
    exit(-1);
  }
}
