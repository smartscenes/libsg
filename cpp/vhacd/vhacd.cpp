#include <fstream>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/tensor.h>
#include <sstream>
#define ENABLE_VHACD_IMPLEMENTATION 1
#include "VHACD.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;
using namespace nb::literals;

using Parameters = VHACD::IVHACD::Parameters;
using nbVecd3 = nb::tensor<nb::numpy, double, nb::shape<nb::any, 3>>;
using nbVecui3 = nb::tensor<nb::numpy, uint32_t, nb::shape<nb::any, 3>>;

class ConvexHull : public VHACD::IVHACD::ConvexHull {
public:
  nbVecd3 getVertices() {
    const size_t shape[2] = {m_points.size(), 3};
    return nbVecd3(&m_points[0], 2, shape);
  }
  nbVecui3 getIndices() {
    const size_t shape[2] = {m_triangles.size(), 3};
    return nbVecui3(&m_triangles[0], 2, shape);
  }
  nbVecd3 getCentroid() {
    const size_t shape[1] = {3};
    return nbVecd3(&m_center, 1, shape);
  }
  nbVecd3 getBBoxMin() {
    const size_t shape[1] = {3};
    return nbVecd3(&mBmin, 1, shape);
  }
  nbVecd3 getBBoxMax() {
    const size_t shape[1] = {3};
    return nbVecd3(&mBmax, 1, shape);
  }
};

class VHACDFacade {
public:
  VHACDFacade() { impl_ = VHACD::CreateVHACD(); }
  ~VHACDFacade() { impl_->Release(); }
  void clean() { return impl_->Clean(); }
  uint32_t numConvexHulls() const { return impl_->GetNConvexHulls(); }
  void getConvexHull(uint32_t i, ConvexHull &c) const {
    impl_->GetConvexHull(i, c);
  }
  std::vector<ConvexHull> getConvexHulls() const {
    std::vector<ConvexHull> chulls;
    for (int i = 0; i < numConvexHulls(); i++) {
      ConvexHull c;
      impl_->GetConvexHull(i, c);
      chulls.push_back(c);
    }
    return chulls;
  }
  bool compute(nb::tensor<double, nb::shape<nb::any>, nb::c_contig> verts,
               nb::tensor<uint32_t, nb::shape<nb::any>, nb::c_contig> tris,
               const Parameters &params) {
    // see https://github.com/wjakob/nanobind/blob/master/docs/tensor.md
    const double *vData = static_cast<double *>(verts.data());
    const uint32_t vLength = verts.shape(0) / 3;
    const uint32_t *iData = static_cast<uint32_t *>(tris.data());
    const uint32_t iLength = tris.shape(0) / 3;
    return impl_->Compute(vData, vLength, iData, iLength, params);
  }

  void saveAsOBJ(const std::string &outFile) {
    FILE *fph = fopen(outFile.c_str(), "w");
    uint32_t idx = 1;
    for (uint32_t ci = 0; ci < this->numConvexHulls(); ci++) {
      fprintf(fph, "o p%03d\n", ci); // object name for each hull
      ConvexHull ch;
      this->getConvexHull(ci, ch);
      for (const VHACD::Vertex &v : ch.m_points) {
        fprintf(fph, "v %0.5f %0.5f %0.5f\n", v.mX, v.mY, v.mZ);
      }
      for (const VHACD::Triangle &t : ch.m_triangles) {
        fprintf(fph, "f %u %u %u\n", t.mI0 + idx, t.mI1 + idx, t.mI2 + idx);
      }
      idx += uint32_t(ch.m_points.size());
    }
    fclose(fph);
  }

private:
  VHACD::IVHACD *impl_;
};

NB_MODULE(vhacd_ext, m) {
  nb::class_<VHACDFacade> vhacd(m, "VHACD");
  vhacd.def(nb::init<>())
      .def("clean", &VHACDFacade::clean)
      .def("compute", &VHACDFacade::compute)
      .def("get_convex_hull", &VHACDFacade::getConvexHull)
      .def("get_convex_hulls", &VHACDFacade::getConvexHulls)
      .def("save_obj", &VHACDFacade::saveAsOBJ)
      .def_property_readonly("num_convex_hulls", &VHACDFacade::numConvexHulls);

  nb::class_<Parameters>(m, "VHACDParams")
      .def("__init__",
           [](Parameters *p) {
             new (p) Parameters();
             p->m_asyncACD = false;
             p->m_fillMode = VHACD::FillMode::FLOOD_FILL;
             p->m_findBestPlane = false;
             p->m_maxConvexHulls = 32;
             p->m_maxNumVerticesPerCH = 8;
             p->m_maxRecursionDepth = 14;
             p->m_minEdgeLength = 2;
             p->m_minimumVolumePercentErrorAllowed = 10.0;
             p->m_resolution = 40000;
             p->m_shrinkWrap = true;
           })
      .def_readwrite("max_hulls", &Parameters::m_maxConvexHulls,
                     "Max convex hulls (32)")
      .def_readwrite("max_hull_verts", &Parameters::m_maxNumVerticesPerCH,
                     "Max vertices per hull (8)")
      .def_readwrite("max_recursion", &Parameters::m_maxRecursionDepth,
                     "Max recursion depth (14)")
      .def_readwrite("num_voxels", &Parameters::m_resolution,
                     "Number of voxels (40K)")
      .def_readwrite("vol_err_perc",
                     &Parameters::m_minimumVolumePercentErrorAllowed,
                     "Volume error allowed in percent (10%)")
      .def_readwrite("min_vox_edge_len", &Parameters::m_minEdgeLength,
                     "Min voxel edge length in voxels (2)")
      .def_readwrite("fill_mode", &Parameters::m_fillMode,
                     "Fill mode (flood), surface or raycast")
      .def_readwrite("shrink_wrap", &Parameters::m_shrinkWrap,
                     "Whether to push vertices to mesh surface (True)")
      .def_readwrite("find_best_plane", &Parameters::m_findBestPlane,
                     "Whether to find best split plane (False)");

  nb::enum_<VHACD::FillMode>(vhacd, "FillMode")
      .value("FloodFill", VHACD::FillMode::FLOOD_FILL)
      .value("RaycastFill", VHACD::FillMode::RAYCAST_FILL)
      .value("Surface", VHACD::FillMode::SURFACE_ONLY)
      .export_values();

  nb::class_<ConvexHull>(vhacd, "ConvexHull")
      .def(nb::init<>())
      .def_readonly("volume", &ConvexHull::m_volume)
      .def_readonly("id", &ConvexHull::m_meshId)
      .def_property_readonly("vertices", &ConvexHull::getVertices)
      .def_property_readonly("indices", &ConvexHull::getIndices)
      .def_property_readonly("center", &ConvexHull::getCentroid)
      .def_property_readonly("min", &ConvexHull::getBBoxMin)
      .def_property_readonly("max", &ConvexHull::getBBoxMax);

#ifdef VERSION
  m.attr("__version__") = MACRO_STRINGIFY(VERSION);
#else
  m.attr("__version__") = "dev";
#endif
}
