/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;



class Step3
{
public:
  Step3();

  void run();


private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


Step3::Step3()
  : fe(1)
  , dof_handler(triangulation)
{}



void Step3::make_grid()
{
  //Maya original
  // GridGenerator::hyper_cube(triangulation, -1, 1);
  // triangulation.refine_global(5);

  //Maya channel_with_cylinder
  // const double shell_region_width = 0.03;
  // const unsigned int n_shells = 2;
  // const double skewness = 2.0;
  // const bool colorize = false;
  //
  // GridGenerator::channel_with_cylinder(triangulation,
  //   shell_region_width, n_shells, skewness, colorize);
  //   triangulation.refine_global(2);

  //Maya personalizada

  // We begin by setting up a grid that is 4 by 22 cells. While not
  // squares, these have pretty good aspect ratios.

  //Parámetros
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
   }






  // std::cout << "Number of active cells: " << bulk_tria.n_active_cells()
  //           << std::endl;

  std::ofstream out2("2_tria_without_cylinder.vtk");
  GridOut       grid_out2;
  grid_out2.write_vtk(tria_without_cylinder, out2);
  std::cout << "Grid written to tria_without_cylinder.vtk" << std::endl;

  std::ofstream out3("3_cylinder_tria.vtk");
  GridOut       grid_out3;
  grid_out3.write_vtk(cylinder_tria, out3);
  std::cout << "Grid written to cylinder_tria.vtk" << std::endl;

  std::ofstream out4("4_shell_tria.vtk");
  GridOut       grid_out4;
  grid_out4.write_vtk(shell_tria, out4);
  std::cout << "Grid written to shell_tria.vtk" << std::endl;

}




void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



void Step3::assemble_system()
{
  QGauss<2> quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1 *                                 // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}



void Step3::solve()
{
  SolverControl solver_control(1000, 1e-12);
  SolverCG<> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}



void Step3::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("solution.gpl");
  data_out.write_gnuplot(output);
}



void Step3::run()
{
  make_grid();
  // setup_system();
  // assemble_system();
  // solve();
  // output_results();
}



int main()
{
  deallog.depth_console(2);

  Step3 laplace_problem;
  laplace_problem.run();

  return 0;
}
