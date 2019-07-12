#ifndef __UTILS_H
#define __UTILS_H
#include <iostream>
#include <errno.h>
#include <sys/stat.h>

void createTreePath(const std::string& path);
bool dirExists(const char* const path);

#endif
