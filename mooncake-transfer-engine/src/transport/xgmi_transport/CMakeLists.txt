file(GLOB XGMI_SOURCES "*.cpp")

add_library(xgmi_transport OBJECT ${XGMI_SOURCES})
target_include_directories(xgmi_transport PUBLIC "${ROCM_PATH}/include")
