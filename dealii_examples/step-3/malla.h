#ifndef malla_h
#define malla_h

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

DEAL_II_NAMESPACE_OPEN

namespace malla
{
  void malla_personal();

  template <>
  void malla_personal2(Triangulation<2> &);
} // namespace malla

#endif
