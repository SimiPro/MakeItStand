#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

// choose exact integral type
#include <CGAL/Gmpzf.h>
typedef CGAL::Gmpzf ET;
//#else
//#include <CGAL/MP_Float.h>
//typedef CGAL::MP_Float ET;
//#endif
//#include <CGAL/Gmpq.h>
//typedef CGAL::Gmpq ET;

// program and solution types

typedef CGAL::Quadratic_program<ET> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

bool DEBUG_OPTIM = false;

// round up to next integer double
double ceil_to_double(const CGAL::Quotient<ET>& x) {
  double a = std::ceil(CGAL::to_double(x));
  while (a < x) a += 1;
  while (a-1 >= x) a -= 1;
  return a;
}

double EPS = 0.1;

// s10_all - s10_empty = 0 => s10_all = s10_empty

// optimize boxes
// beta[i] = 1 empty box
// beta[i] = 0 fill box
void optim(const VectorXd &b_s10_all, const vector<VectorXd> &b_s10_interior, VectorXd &betas) {
    cout << "Start hollowing calculation.. " << endl;

    betas.resize(b_s10_interior.size()); betas.setZero();

    // 0 <= betas <= 1
    Program lp (CGAL::EQUAL, true, 0.0, true, 1.0);

    // min sy'.betas
    for (int i = 0; i < b_s10_interior.size(); i++) {
        lp.set_c(i, -(b_s10_interior[i][2]*b_s10_interior[i][2])); // b_s10[i][2] = sy
    }
    lp.set_c0(b_s10_all[2]);

    // subj to sx = sz = 0
    int SX = 0;
    int SZ = 1;
    for (int i = 0; i < b_s10_interior.size(); i++) {
        lp.set_a(i, SX, b_s10_interior[i][1]); // b_s10_interior[i][1] = sx'
        lp.set_a(i, SZ, b_s10_interior[i][3]); // b_s10_interior[i][3] = sz'
    }
    lp.set_b(SX, b_s10_all[1]);
    lp.set_b(SZ, b_s10_all[3]);

    Solution s = CGAL::solve_linear_program(lp, ET());
    assert (s.solves_linear_program(lp));
    cout << "LP Solution: " << endl;
    cout << s << endl;

    if (s.is_infeasible()) {
        cout << "Attention! Deform the mesh! LP has no solution" << endl;
    }

    //cout <<  ceil_to_double(s.objective_value()) << endl;
   // cout << "objective value: " << s.objective_value() << endl;
   // cout << s << endl;

    auto it = s.variable_values_begin();
    auto end = s.variable_values_end();
    int counter = 0;
    int empty_counter = 0;
    int non_empty_counter = 0;
    for (; it != end; ++it) {
        if (DEBUG_OPTIM)
            cout << "var: " << counter << " | " << CGAL::to_double(*it) << " ";
        double val = ceil_to_double(*it);
        //cout << "var: " << counter << " " << val << endl;
        if (val < EPS) { // fill this shizzl
            betas[counter] = 0;
            non_empty_counter++;
        } else { // val > 1 - EPS -> empty this shizzle
            betas[counter] = 1;
            empty_counter++;
        }
        counter++;
    }
    std::cout << std::endl;
    
    //cout << "betas: " << endl;
    //cout << betas << endl;

    std::cout << "Empty boxes: " << empty_counter << " | filled boxes: " << non_empty_counter << std::endl;


}
