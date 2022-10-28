#ifndef BASE_WEAVER_H
#define BASE_WEAVER_H

#include <string>

class BaseWeaver {
public:
    virtual float weaveIteration() = 0;
    virtual void saveCurrentImage(const char* imagePath) = 0;
    virtual std::string getInstructionsStr() = 0;
};

#endif