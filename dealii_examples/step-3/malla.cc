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

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <vector>

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
    const bool colorize = false;
    Triangulation<2> tria;

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
     }
    GridTools::shift(cylinder_triangulation_offset, cylinder_tria);

    // Compute the tolerance again, since the shells may be very close to
    // each-other:
    const double vertex_tolerance =
      std::min(internal::minimal_vertex_distance(tria_without_cylinder),
               internal::minimal_vertex_distance(cylinder_tria)) /
      10;
    GridGenerator::merge_triangulations(
      tria_without_cylinder, cylinder_tria, tria, vertex_tolerance, true);

    // Ensure that all manifold ids on a polar cell really are set to the
    // polar manifold id:
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->manifold_id() == polar_manifold_id)
        cell->set_all_manifold_ids(polar_manifold_id);

    // Ensure that all other manifold ids (including the interior faces
    // opposite the cylinder) are set to the flat manifold id:
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->manifold_id() != polar_manifold_id &&
          cell->manifold_id() != tfi_manifold_id)
        cell->set_all_manifold_ids(numbers::flat_manifold_id);

    // We need to calculate the current center so that we can move it later:
    // to start get a unique list of (points to) vertices on the cylinder
    std::vector<Point<2> *> cylinder_pointers;
    for (const auto &face : tria.active_face_iterators())
      if (face->manifold_id() == polar_manifold_id)
        {
          cylinder_pointers.push_back(&face->vertex(0));
          cylinder_pointers.push_back(&face->vertex(1));
        }
    // de-duplicate
    std::sort(cylinder_pointers.begin(), cylinder_pointers.end());
    cylinder_pointers.erase(std::unique(cylinder_pointers.begin(),
                                        cylinder_pointers.end()),
                            cylinder_pointers.end());

    // find the current center...
    Point<2> center;
    for (const Point<2> *const ptr : cylinder_pointers)
      center += *ptr / double(cylinder_pointers.size());

    // and recenter at (0.2, 0.2)
    for (Point<2> *const ptr : cylinder_pointers)
      *ptr += Point<2>(0.2, 0.2) - center;

    // attach manifolds
    PolarManifold<2> polar_manifold(Point<2>(0.2, 0.2));
    tria.set_manifold(polar_manifold_id, polar_manifold);
    TransfiniteInterpolationManifold<2> inner_manifold;
    inner_manifold.initialize(tria);
    tria.set_manifold(tfi_manifold_id, inner_manifold);

    if (colorize)
      for (const auto &face : tria.active_face_iterators())
        if (face->at_boundary())
          {
            const Point<2> center = face->center();
            // left side
            if (std::abs(center[0] - 0.0) < 1e-10)
              face->set_boundary_id(0);
            // right side
            else if (std::abs(center[0] - 2.2) < 1e-10)
              face->set_boundary_id(1);
            // cylinder boundary
            else if (face->manifold_id() == polar_manifold_id)
              face->set_boundary_id(2);
            // sides of channel
            else
              {
                Assert(std::abs(center[1] - 0.00) < 1.0e-10 ||
                         std::abs(center[1] - 0.41) < 1.0e-10,
                       ExcInternalError());
                face->set_boundary_id(3);
              }
          }

    // std::ofstream out("8_final_tria.vtk");
    // GridOut       grid_out;
    // grid_out.write_vtk(tria, out);
    // std::cout << "Grid written to 8_final_tria.vtk" << std::endl;
  }

  void malla_personal2()
  {
    //Parameters
    // const types::manifold_id polar_manifold_id = 0;
    // const types::manifold_id tfi_manifold_id   = 1;
    // const double l_bulk = 1000.0;
    // const unsigned int n_wells = 2; //number of wells
    // std::vector< std::vector<double> > well_loc[n_wells-1];
    //std::vector< Point<2> > well_loc[n_wells-1];
    const std::vector<double> well_loc_1 = {500.0,500.0};
    const std::vector<double> well_loc_2 = {800.0,800.0};
    // std::vector<std::vector<double> > well_loc{ { 1.5, 2.0, 3.0 },
    //                                             { 4.0, 5.0, 6.0 },
    //                                             { 7.0, 8.0, 9.0 } };
    std::vector<std::vector<double> > well_loc;
    // const Point<2> well_loc_1(500.0, 500.0); //well location
    // const Point<2> well_loc_2(800.0, 800.0); //well location
    // const unsigned int n_cells_bulk = 10;
    // const unsigned int n_cells_r = 10;
    // const unsigned int n_cells_tet = 8;
    // const double rw_well_1 = 0.35; // well radius
    // const double re_well_1 = 100.0; // aprox drainage radius
    // const std::vector<unsigned int> bulk_cells = {n_cells_bulk, n_cells_bulk};
    // const Point<2> bulk_P1(0.0, 0.0);
    // const Point<2> bulk_P2(l_bulk, l_bulk);
    // const double shell_region_width = re_well_1 * 0.8;
    // const double cyl_inner_radius = rw_well_1 + shell_region_width;
    // const double cyl_outer_radius = re_well_1;
    // const double shell_inner_radius = rw_well_1;
    // const double shell_outer_radius = rw_well_1 + shell_region_width;
    // const unsigned int n_shells = n_cells_r;
    // const double skewness = 2.0;
    // const unsigned int n_cells_per_shell = n_cells_tet;
    // //Tensor<1, 2> cylinder_triangulation_offset = well_loc_1;
    // Triangulation<2> tria;

    well_loc.push_back(well_loc_1);
    well_loc.push_back(well_loc_2);

    // Displaying the 2D vector
    std::cout << well_loc[0][0] << std::endl;
    // for (unsigned int i = 0; i < well_loc.size(); i++)
    //  {
    //     for (unsigned int j = 0; j < well_loc[i].size(); j++)
    //         std::cout << well_loc[i][j] << " ";
    //     std::cout << std::endl;
    //  }

    // //Bulk grid creation
    // Triangulation<2> bulk_tria;
    // GridGenerator::subdivided_hyper_rectangle(bulk_tria,
    //                                           bulk_cells,
    //                                           bulk_P1,
    //                                           bulk_P2);

    // //Cells removing
    // std::set<Triangulation<2>::active_cell_iterator> cells_to_remove;
    // for (const auto &cell : bulk_tria.active_cell_iterators())
    //   {
    //     // Colect the cells to remove, those which center is inside
    //     // the square re_well_1 x re_well_1
    //     // if ((std::fabs((cell->center()[0] - well_loc_1[0])) < re_well_1) &&
    //     //      (std::fabs((cell->center()[1] - well_loc_1[1])) < re_well_1 ))
    //     //        cells_to_remove.insert(cell);
    //     if ((std::fabs((cell->center()[0] - well_loc[0][0])) < re_well_1) &&
    //         (std::fabs((cell->center()[1] - well_loc[0][1])) < re_well_1 ))
    //           cells_to_remove.insert(cell);
    //   }

    // //Create the grid with removed cells
    // Triangulation<2> tria_without_cylinder;
    // GridGenerator::create_triangulation_with_removed_cells(
    //   bulk_tria, cells_to_remove, tria_without_cylinder);
    //
    // std::ofstream out("15_well_loc.vtk");
    // GridOut       grid_out;
    // grid_out.write_vtk(tria_without_cylinder, out);
    // std::cout << "Grid written to 15_well_loc.vtk" << std::endl;
    // // set up the cylinder triangulation. Note that this function sets the
    // // manifold ids of the interior boundary cells to 0
    // // (polar_manifold_id).
    // Triangulation<2> cylinder_tria;
    // GridGenerator::hyper_cube_with_cylindrical_hole(cylinder_tria,
    //                                                 cyl_inner_radius,
    //                                                 cyl_outer_radius);
    //
    // // Assign interior manifold ids to be the TFI id.
    // for (const auto &cell : cylinder_tria.active_cell_iterators())
    //  {
    //    cell->set_manifold_id(tfi_manifold_id);
    //    for (unsigned int face_n = 0; face_n < GeometryInfo<2>::faces_per_cell;
    //         ++face_n)
    //      if (!cell->face(face_n)->at_boundary())
    //        cell->face(face_n)->set_manifold_id(tfi_manifold_id);
    //  }
    // if (0.0 < shell_region_width)
    //  {
    //    Assert(0 < n_shells,
    //           ExcMessage("If the shell region has positive width then "
    //                      "there must be at least one shell."));
    //    Triangulation<2> shell_tria;
    //    GridGenerator::concentric_hyper_shells(shell_tria,
    //                                           Point<2>(),
    //                                           shell_inner_radius,
    //                                           shell_outer_radius,
    //                                           n_shells,
    //                                           skewness,
    //                                           n_cells_per_shell);
    //
    //    // Make the tolerance as large as possible since these cells can
    //    // be quite close together
    //    const double vertex_tolerance =
    //      std::min(internal::minimal_vertex_distance(shell_tria),
    //               internal::minimal_vertex_distance(cylinder_tria)) *
    //      0.5;
    //
    //    shell_tria.set_all_manifold_ids(polar_manifold_id);
    //    Triangulation<2> temp;
    //    GridGenerator::merge_triangulations(
    //      shell_tria, cylinder_tria, temp, vertex_tolerance, true);
    //    cylinder_tria = std::move(temp);
    //  }
    //
    // GridTools::shift(cylinder_triangulation_offset, cylinder_tria);
    //
    // // Compute the tolerance again, since the shells may be very close to
    // // each-other:
    // const double vertex_tolerance =
    //   std::min(internal::minimal_vertex_distance(tria_without_cylinder),
    //            internal::minimal_vertex_distance(cylinder_tria)) /
    //   10;
    //
    // GridGenerator::merge_triangulations(
    //   tria_without_cylinder, cylinder_tria, tria, vertex_tolerance, true);
    //
    // // Ensure that all manifold ids on a polar cell really are set to the
    // // polar manifold id:
    // for (const auto &cell : tria.active_cell_iterators())
    //   if (cell->manifold_id() == polar_manifold_id)
    //     cell->set_all_manifold_ids(polar_manifold_id);
    //
    // // Ensure that all other manifold ids (including the interior faces
    // // opposite the cylinder) are set to the flat manifold id:
    // for (const auto &cell : tria.active_cell_iterators())
    //   if (cell->manifold_id() != polar_manifold_id &&
    //       cell->manifold_id() != tfi_manifold_id)
    //     cell->set_all_manifold_ids(numbers::flat_manifold_id);
    //
    // // attach manifolds
    // PolarManifold<2> polar_manifold(well_loc_1);
    // tria.set_manifold(polar_manifold_id, polar_manifold);
    // TransfiniteInterpolationManifold<2> inner_manifold;
    // inner_manifold.initialize(tria);
    // tria.set_manifold(tfi_manifold_id, inner_manifold);
    //
    // // if (colorize)
    // //   for (const auto &face : tria.active_face_iterators())
    // //     if (face->at_boundary())
    // //       {
    // //         const Point<2> center = face->center();
    // //         // left side
    // //         if (std::abs(center[0] - 0.0) < 1e-10)
    // //           face->set_boundary_id(0);
    // //         // right side
    // //         else if (std::abs(center[0] - 2.2) < 1e-10)
    // //           face->set_boundary_id(1);
    // //         // cylinder boundary
    // //         else if (face->manifold_id() == polar_manifold_id)
    // //           face->set_boundary_id(2);
    // //         // sides of channel
    // //         else
    // //           {
    // //             Assert(std::abs(center[1] - 0.00) < 1.0e-10 ||
    // //                      std::abs(center[1] - 0.41) < 1.0e-10,
    // //                    ExcInternalError());
    // //             face->set_boundary_id(3);
    // //           }
    // //       }
    //
    // // std::ofstream out("14_mi_merged_tria.vtk");
    // // GridOut       grid_out;
    // // grid_out.write_vtk(tria, out);
    // // std::cout << "Grid written to 14_mi_merged_tria.vtk" << std::endl;

  }
}
