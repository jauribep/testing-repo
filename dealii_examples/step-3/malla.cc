#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <cmath>
#include <limits>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include "malla.h"

using namespace dealii;

namespace malla
{
  //Funciones
  namespace internal
  {
    // Find the minimal distance between two vertices. This is useful for
    // computing a tolerance for merging vertices in
    // GridTools::merge_triangulations.
    template <int dim, int spacedim>
    double
    minimal_vertex_distance(const Triangulation<dim, spacedim> &triangulation)
    {
      double length = std::numeric_limits<double>::max();
      for (const auto &cell : triangulation.active_cell_iterators())
        for (unsigned int n = 0; n < GeometryInfo<dim>::lines_per_cell; ++n)
          length = std::min(length, cell->line(n)->diameter());
      return length;
    }
  } // namespace internal

  void malla_personal()
  {
    //Parámetros
    const types::manifold_id polar_manifold_id = 0;
    const types::manifold_id tfi_manifold_id   = 1;
    const std::vector<unsigned int> bulk_cells = {22u, 4u};
    const Point<2> bulk_P1(0.0, 0.0);
    const Point<2> bulk_P2(2.2, 0.41);
    const double shell_region_width = 0.03;
    const double cyl_inner_radius = 0.05 + shell_region_width;
    const double cyl_outer_radius = 0.41 / 4.0;
    const double shell_inner_radius = 0.05;
    const double shell_outer_radius = 0.05 + shell_region_width;
    const unsigned int n_shells = 2;
    const double skewness = 2.0;
    const unsigned int n_cells_per_shell = 8;

    //Construcción de la malla
    Triangulation<2> bulk_tria;
    GridGenerator::subdivided_hyper_rectangle(bulk_tria,
                                              bulk_cells,
                                              bulk_P1,
                                              bulk_P2);
    // bulk_tria now looks like this:
    //
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |XX|XX|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--O--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |XX|XX|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //
    // Note that these cells are not quite squares: they are all 0.1 by
    // 0.1025.
    //
    // The next step is to remove the cells marked with XXs: we will place
    // the grid around the cylinder there later. The next loop does two
    // things:
    // 1. Determines which cells need to be removed from the Triangulation
    //    (i.e., find the cells marked with XX in the picture).
    // 2. Finds the location of the vertex marked with 'O' and uses that to
    //    calculate the shift vector for aligning cylinder_tria with
    //    tria_without_cylinder.
    std::set<Triangulation<2>::active_cell_iterator> cells_to_remove;
    Tensor<1, 2> cylinder_triangulation_offset;
    for (const auto &cell : bulk_tria.active_cell_iterators())
      {
        if ((cell->center() - Point<2>(0.2, 0.2)).norm() < 0.15)
          cells_to_remove.insert(cell);

        if (cylinder_triangulation_offset == Tensor<1, 2>())
          {
            for (unsigned int vertex_n = 0;
                 vertex_n < GeometryInfo<2>::vertices_per_cell;
                 ++vertex_n)
              if (cell->vertex(vertex_n) == Point<2>())
                {
                  // cylinder_tria is centered at zero, so we need to
                  // shift it up and to the right by two cells:
                  cylinder_triangulation_offset =
                    2.0 * (cell->vertex(3) - Point<2>());
                  break;
                }
          }
      }
    Triangulation<2> tria_without_cylinder;
    GridGenerator::create_triangulation_with_removed_cells(
      bulk_tria, cells_to_remove, tria_without_cylinder);

    // set up the cylinder triangulation. Note that this function sets the
    // manifold ids of the interior boundary cells to 0
    // (polar_manifold_id).
    Triangulation<2> cylinder_tria;
    GridGenerator::hyper_cube_with_cylindrical_hole(cylinder_tria,
                                                    cyl_inner_radius,
                                                    cyl_outer_radius);
    // The bulk cells are not quite squares, so we need to move the left
    // and right sides of cylinder_tria inwards so that it fits in
    // bulk_tria:
    for (const auto &cell : cylinder_tria.active_cell_iterators())
      for (unsigned int vertex_n = 0;
           vertex_n < GeometryInfo<2>::vertices_per_cell;
           ++vertex_n)
        {
          if (std::abs(cell->vertex(vertex_n)[0] - -0.41 / 4.0) < 1e-10)
            cell->vertex(vertex_n)[0] = -0.1;
          else if (std::abs(cell->vertex(vertex_n)[0] - 0.41 / 4.0) < 1e-10)
            cell->vertex(vertex_n)[0] = 0.1;
        }

    // Assign interior manifold ids to be the TFI id.
    for (const auto &cell : cylinder_tria.active_cell_iterators())
     {
       cell->set_manifold_id(tfi_manifold_id);
       for (unsigned int face_n = 0; face_n < GeometryInfo<2>::faces_per_cell;
            ++face_n)
         if (!cell->face(face_n)->at_boundary())
           cell->face(face_n)->set_manifold_id(tfi_manifold_id);
     }
    if (0.0 < shell_region_width)
     {
       Assert(0 < n_shells,
              ExcMessage("If the shell region has positive width then "
                         "there must be at least one shell."));
       Triangulation<2> shell_tria;
       GridGenerator::concentric_hyper_shells(shell_tria,
                                              Point<2>(),
                                              shell_inner_radius,
                                              shell_outer_radius,
                                              n_shells,
                                              skewness,
                                              n_cells_per_shell);

       // Make the tolerance as large as possible since these cells can
       // be quite close together
       const double vertex_tolerance =
         std::min(internal::minimal_vertex_distance(shell_tria),
                  internal::minimal_vertex_distance(cylinder_tria)) *
         0.5;

       shell_tria.set_all_manifold_ids(polar_manifold_id);
       Triangulation<2> temp;
       GridGenerator::merge_triangulations(
         shell_tria, cylinder_tria, temp, vertex_tolerance, true);
       cylinder_tria = std::move(temp);

       std::ofstream out("5_cylinder_tria2.vtk");
       GridOut       grid_out;
       grid_out.write_vtk(cylinder_tria, out);
       std::cout << "Grid written to cylinder_tria2.vtk" << std::endl;


     }
  }
}
