#ifndef malla_h
#define malla_h

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

namespace malla
{
  void malla_personal();
  void malla_personal2(const Triangulation<2> &tria);
} // namespace malla

#endif
