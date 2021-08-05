#ifndef malla_h
#define malla_h

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

namespace malla
{
  void malla_personal();

  void malla_personal2(Triangulation<2> &triangulation);
} // namespace malla

#endif
