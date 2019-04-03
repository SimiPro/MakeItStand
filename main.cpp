#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <vector>

#include <igl/readOBJ.h>
#include <igl/unproject_onto_mesh.h>

#include <igl/centroid.h>


using namespace std;

typedef igl::opengl::glfw::Viewer Viewer;

Eigen::MatrixXd V, C;
Eigen::MatrixXd N;
Eigen::MatrixXi F;



Eigen::Vector3d gravity;
vector<Eigen::Vector3d> handles;

bool set_gravity = false;
bool set_handles = false;



bool mouse_down(Viewer& viewer, int button, int modifier) {
    if (set_handles) {
        int fid;
        Eigen::Vector3f bc;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core.viewport(3) - viewer.current_mouse_y;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
          viewer.core.proj, viewer.core.viewport, V, F, fid, bc)) {
          // paint hit red
          C.row(fid)<<1,0,0;
          viewer.data().set_colors(C);
          return false;
        }
    }
    return false;
}   

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        Eigen::RowVector3d cen;
        double vol;

	//std::cout << "Start: compute center of mass" << std::endl;
        igl::centroid(V, F, cen, vol); 
	//std::cout << "Done" << std::endl;

        viewer.data().add_points(cen, Eigen::RowVector3d(1,0,0));
    }
}



int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ./bin <mesh.off>" << endl;
        igl::readOFF("../data/sphere.off",V,F,N);
    } else {
        // Read points and normals
        igl::readOFF(argv[1],V,F,N);
    }
    C = Eigen::MatrixXd::Constant(F.rows(),3,1);

    // init viewer
    Viewer viewer;


    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_custom_window = [&]() {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiSetCond_FirstUseEver);
        ImGui::Begin(
            "Window Handling Stuff", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );   

        ImGui::Checkbox("Set Gravity", &set_gravity);

        ImGui::Checkbox("Set Handles", &set_handles);



        ImGui::End();
    };

    
    viewer.callback_mouse_down = &mouse_down;
    viewer.callback_key_down = callback_key_down;

    // Plot the mesh

    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
