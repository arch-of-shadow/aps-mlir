message(STATUS "Building TORSCHEDULE with additional header directories: ${PROJECT_SOURCE_DIR}/include/TOR and ${PROJECT_SOURCE_DIR}/include/Schedule")

add_mlir_dialect_library(TORSCHEDULE
    ScheduleAlgo.cpp
    SDCSolver.cpp
    SDCSchedule.cpp
    CDFG.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TOR
    ${RROJECT_SOURCE_DIR}/include/Schedule

    LINK_LIBS PUBLIC
    MLIRIR
)

if (LP_SOLVE_INCLUDE_DIR)
    target_include_directories(TORSCHEDULE PUBLIC ${LP_SOLVE_INCLUDE_DIR})
endif ()

if (LP_SOLVE_LIB)
    target_link_libraries(TORSCHEDULE PUBLIC ${LP_SOLVE_LIB})
endif ()
