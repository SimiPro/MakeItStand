#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

// choose exact integral type
//#include <CGAL/Gmpzf.h>
//typedef CGAL::Gmpzf ET;
//#else
#include <CGAL/MP_Float.h>
typedef CGAL::MP_Float ET;
//#endif
//#include <CGAL/Gmpq.h>
//typedef CGAL::Gmpq ET;

// program and solution types

typedef CGAL::Quadratic_program<ET> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

// round up to next integer double
double ceil_to_double(const CGAL::Quotient<ET>& x) {
  double a = std::ceil(CGAL::to_double(x));
  while (a < x) a += 1;
  while (a-1 >= x) a -= 1;
  return a;
}

double EPS = 1e-3;

void optim(const VectorXd &b_s10_all, const vector<VectorXd> &b_s10_interior, VectorXd &betas) {
    cout << "Start hollowing calculation.. " << endl;

    betas.resize(b_s10_interior.size()); betas.setZero();

    // 0 <= betas <= 1
    Program lp (CGAL::EQUAL, true, 0, true, 1);

    // min sy'.betas
    for (int i = 0; i < b_s10_interior.size(); i++) {
        lp.set_c(i, -b_s10_interior[i][2]); // b_s10[i][2] = sy
    }
    lp.set_c0(b_s10_all[2]);

    // subj to sx = sz = 0
    int SX = 0;
    int SZ = 1;
    for (int i = 0; i < b_s10_interior.size(); i++) {
        lp.set_a(i, SX, b_s10_interior[i][1]); // b_s10_interior[i][1] = sx'
        lp.set_a(i, SZ, b_s10_interior[i][3]); // b_s10_interior[i][1] = sz'
    }
    lp.set_b(SX, b_s10_all[1]);
    lp.set_b(SZ, b_s10_all[3]);

    Solution s = CGAL::solve_linear_program(lp, ET());
    assert (s.solves_linear_program(lp));

    //cout <<  ceil_to_double(s.objective_value()) << endl;
   // cout << "objective value: " << s.objective_value() << endl;
   // cout << s << endl;

    Solution::Index_iterator it = s.basic_variable_indices_begin();
    Solution::Index_iterator end = s.basic_variable_indices_end();
    int counter = 0;
    for (; it != end; ++it) {
        double val = ceil_to_double(*it);
        if (val < EPS) { // hollow this shizzl
            betas[counter] = 0;
        } else {
            betas[counter] = 1;
        }
        counter++;
    }
    std::cout << std::endl;
    
    cout << "betas: " << endl;
    cout << betas << endl;

}
