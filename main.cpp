#include <QGuiApplication>
#include <QSurfaceFormat>
#include "window.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setVersion(3,3);

    Window window;
    window.setFormat(format);
    window.resize(QSize(800,800));
    window.show();//FullScreen();

    return app.exec();
}
