if(DEFINED ENV{CONDA_PREFIX})
  set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

find_package(PkgConfig REQUIRED)

if(DEFINED ENV{CONDA_PREFIX})
  set(PKG_CONFIG_ARGN "--define-variable=PREFIX=$ENV{CONDA_PREFIX}")
endif()

set(Cbc_NO_Cbc_CMAKE ON CACHE BOOL "Use pixi pkg-config Cbc instead of CMake packages" FORCE)
set(Clp_NO_Clp_CMAKE ON CACHE BOOL "Use pixi pkg-config Clp instead of CMake packages" FORCE)

if(NOT TARGET PkgConfig::Cbc)
  pkg_check_modules(Cbc REQUIRED IMPORTED_TARGET GLOBAL cbc)
endif()

if(NOT TARGET PkgConfig::OsiCbc)
  pkg_check_modules(OsiCbc REQUIRED IMPORTED_TARGET GLOBAL osi-cbc)
endif()

if(NOT TARGET PkgConfig::Clp)
  pkg_check_modules(Clp REQUIRED IMPORTED_TARGET GLOBAL clp)
endif()

if(NOT TARGET PkgConfig::OsiClp)
  pkg_check_modules(OsiClp REQUIRED IMPORTED_TARGET GLOBAL osi-clp)
endif()

if(NOT TARGET PkgConfig::Cgl)
  pkg_check_modules(Cgl REQUIRED IMPORTED_TARGET GLOBAL cgl)
endif()

if(NOT TARGET PkgConfig::Osi)
  pkg_check_modules(Osi REQUIRED IMPORTED_TARGET GLOBAL osi)
endif()

if(NOT TARGET PkgConfig::CoinUtils)
  pkg_check_modules(CoinUtils REQUIRED IMPORTED_TARGET GLOBAL coinutils)
endif()

function(aps_append_coinor_deps target)
  if(TARGET ${target})
    target_link_libraries(${target} INTERFACE
      PkgConfig::Cgl
      PkgConfig::OsiClp
      PkgConfig::Clp
      PkgConfig::Osi
      PkgConfig::CoinUtils
    )
  endif()
endfunction()

aps_append_coinor_deps(PkgConfig::Cbc)
aps_append_coinor_deps(PkgConfig::OsiCbc)
aps_append_coinor_deps(PkgConfig::Clp)
aps_append_coinor_deps(PkgConfig::OsiClp)
