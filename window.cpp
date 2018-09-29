#include <QCoreApplication>
#include <QDebug>
#include <QString>
#include <QOpenGLShaderProgram>
#include <QExposeEvent>
#include "input.h"
#include "window.h"
#include "vertex.h"

// Create a colored single fullscreen triangle
// 3*______________
//  |\_____________
//  | \____________
// 2*  \___________
//  |   \__________
//  |    \_________
// 1*--*--*________
//  |.....|\_______
//  |.....| \______
// 0*..*..*  \_____
//  |.....|   \____
//  |.....|    \___
//-1*--*--*--*--*__
// -1  0  1  2  3 x position coords
//  0     1     2 s texture coords
static const Vertex sg_vertexes[] = {
    Vertex( QVector3D(-1.0f,  3.0f, -1.0f), QVector3D(2.0f, 0.0f, 0.0f) ),
    Vertex( QVector3D(-1.0f, -1.0f, -1.0f), QVector3D(0.0f, 0.0f, 1.0f) ),
    Vertex( QVector3D( 3.0f, -1.0f, -1.0f), QVector3D(0.0f, 2.0f, 0.0f) )
};

Window::Window()
{

}

Window::~Window()
{
    makeCurrent();
    teardownGL();
}

// ======================================================================
// public
// ======================================================================

void Window::initializeGL()
{
    // Initialize OpenGL Backend
    initializeOpenGLFunctions();
    connect(this, SIGNAL(frameSwapped()), this, SLOT(update()));
    printInfo();

    // Set global information
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Application-specific initialization
    {
        // Create Shader (Do not release until VAO is created)
        m_program = new QOpenGLShaderProgram();
        m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/basic_vert.glsl");
        m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/basic_frag.glsl");
        m_program->link();
        m_program->bind();

        // Create Buffer (Do not release until VAO is created)
        m_vertex.create();
        m_vertex.bind();
        m_vertex.setUsagePattern(QOpenGLBuffer::StaticDraw);
        m_vertex.allocate(sg_vertexes, sizeof(sg_vertexes));

        // Create Vertex Array Object
        m_object.create();
        m_object.bind();
        m_program->enableAttributeArray(0);
        m_program->enableAttributeArray(1);
        m_program->setAttributeBuffer(0, GL_FLOAT, Vertex::positionOffset(), Vertex::PositionTupleSize, Vertex::stride());
        m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());

        // Release (unbind) all
        m_object.release();
        m_vertex.release();
        m_program->release();
    }
}

void Window::resizeGL(int width, int height)
{
    // Currently we are not handling width/height changes
    (void)width;
    (void)height;
}

void Window::paintGL()
{
    // Clear
    glClear(GL_COLOR_BUFFER_BIT);

    // Render using our shader
    m_program->bind();
    {
        m_object.bind();
        glDrawArrays(GL_TRIANGLES, 0, sizeof(sg_vertexes) / sizeof(sg_vertexes[0]));
        m_object.release();
    }
    m_program->release();
}

void Window::teardownGL()
{
    // Actually destroy our OpenGL information
    m_object.destroy();
    m_vertex.destroy();
    delete m_program;
}

// ======================================================================
// protected
// ======================================================================

void Window::update()
{
    // Update input
    Input::update();

    if (Input::keyPressed(Qt::Key_Escape)) {
        QCoreApplication::quit();
    }

    QOpenGLWindow::update();
}

void Window::keyPressEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat())
    {
        event->ignore();
    }
    else
    {
        Input::registerKeyPress(event->key());
    }
}

void Window::keyReleaseEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat())
    {
        event->ignore();
    }
    else
    {
        Input::registerKeyRelease(event->key());
    }
}

void Window::mousePressEvent(QMouseEvent *event)
{
    Input::registerMousePress(event->button());
}

void Window::mouseReleaseEvent(QMouseEvent *event)
{
    Input::registerMouseRelease(event->button());
}

// ======================================================================
// private
// ======================================================================

void Window::printInfo()
{
    QString glType;
    QString glVersion;
    QString glProfile;

    // Get Version Information
    glType = (context()->isOpenGLES()) ? "OpenGL ES" : "OpenGL";
    glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));

    // Get Profile Information
#define CASE(c) case QSurfaceFormat::c: glProfile = #c; break
    switch (format().profile())
    {
    CASE(NoProfile);
    CASE(CoreProfile);
    CASE(CompatibilityProfile);
    }
#undef CASE

    // qPrintable() will print our QString w/o quotes around it.
    qDebug() << qPrintable(glType) << qPrintable(glVersion) << "(" << qPrintable(glProfile) << ")";
}
