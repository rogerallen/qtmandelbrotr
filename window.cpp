#include <QCoreApplication>
#include <QDebug>
#include <QString>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QExposeEvent>
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "input.h"
#include "window.h"
#include "vertex.h"
#include "cuda_utils.h"

extern void mandelbrot(void *devPtr, int w, int h, double cx, double cy, double zoom);

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
    Vertex( QVector3D(-1.0f,  3.0f, -1.0f), QVector3D(0.0f, 2.0f, 0.0f) ),
    Vertex( QVector3D(-1.0f, -1.0f, -1.0f), QVector3D(0.0f, 0.0f, 1.0f) ),
    Vertex( QVector3D( 3.0f, -1.0f, -1.0f), QVector3D(2.0f, 0.0f, 0.0f) )
};

Window::Window()
{
    m_shared_width = m_shared_height = 2048;
    m_center_x = 0.0;
    m_center_y = 0.0;
    m_zoom = 0.5;
    m_is_full_screen = false;
    m_zoom_out_mode = false;
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

    // OpenGL initialization
    {
        // Create Shader (Do not release until VAO is created)
        m_program = new QOpenGLShaderProgram();
        m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/basic_vert.glsl");
        m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/basic_frag.glsl");
        m_program->link();
        m_program->bind();

        // Bind uniforms
        u_cameraToView = m_program->uniformLocation("cameraToView");

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

        // Load texture (not used anymore FIXME?)
        m_texture = new QOpenGLTexture(QImage(QString(":/images/side1.png")).mirrored());

        // Shared OpenGL & CUDA buffer
        // Generate a buffer ID
        glGenBuffers(1,&m_shared_pbo_id);
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_shared_pbo_id);
        // Allocate data for the buffer
        glBufferData(GL_PIXEL_UNPACK_BUFFER, m_shared_width * m_shared_height * 4, nullptr, GL_DYNAMIC_COPY);

        // Create a GL Texture
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1,&m_shared_tex_id);
        glBindTexture(GL_TEXTURE_2D, m_shared_tex_id);
        // Allocate the texture memory.
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, m_shared_width, m_shared_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
        // Set filter mode
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    }

    // CUDA initialization
    {
        cuda_init();
        m_cuda_pbo_handle = cuda_register_buffer(m_shared_pbo_id);
    }
}

// keep the longest side (-1,1) and keep centered around 0,0 origin.
void Window::resizeGL(int width, int height)
{
    m_window_width = width;
    m_window_height = height;
    float aspect = float(width)/float(height);

    if (width >= height) {
        // aspect >= 1, set the width to -1 to 1, with larger height
        m_projection = glm::ortho(-1.0f, 1.0f, -1.0f / aspect, 1.0f / aspect);
    } else {
        // aspect < 1, set the height from -1 to 1, with larger width
        m_projection = glm::ortho(-1.0f * aspect, 1.0f * aspect, -1.0f, 1.0f);
    }
}

void Window::paintGL()
{
    // Clear
    glClear(GL_COLOR_BUFFER_BIT);

    // Do some CUDA that writes to the pbo
    void *devPtr = cuda_map_resource(m_cuda_pbo_handle);
    mandelbrot(devPtr, m_shared_width, m_shared_height, m_center_x, m_center_y, m_zoom);
    cuda_unmap_resource(m_cuda_pbo_handle);

    // Render using our shader
    m_program->bind();
    m_program->setUniformValue(u_cameraToView, QMatrix4x4(glm::value_ptr(m_projection)));
    {
        m_object.bind();

        // the equivalent of:
        // m_texture->bind();
        // connect the pbo to the texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_shared_pbo_id);
        glBindTexture(GL_TEXTURE_2D, m_shared_tex_id);
        // Since source parameter is NULL, Data is coming from a PBO, not host memory
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_shared_width, m_shared_height, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glDrawArrays(GL_TRIANGLES, 0, sizeof(sg_vertexes) / sizeof(sg_vertexes[0]));
        m_object.release();
    }
    m_program->release();
}

void Window::teardownGL()
{
    // Destroy our CUDA info
    cuda_unregister_resource(m_cuda_pbo_handle);

    // Destroy our OpenGL information
    m_object.destroy();
    m_vertex.destroy();
    delete m_program;
    delete m_texture;
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
    else if (Input::keyPressed(Qt::Key_F)) {
        // FIXME -- we get multiple calls & we don't want that.
        if (m_is_full_screen) {
            showNormal();
            m_is_full_screen = false;
        } else {
            showFullScreen();
            m_is_full_screen = true;
        }
    } else if (Input::keyPressed(Qt::Key_Return)) {
        m_zoom_out_mode = true;
        printf("zoom out!\n");
    }

    if(m_zoom_out_mode) {
        m_zoom /= 1.1;
        if(m_zoom < 0.5) {
            m_zoom_out_mode = false;
        }
    }

    if (Input::buttonPressed(Qt::LeftButton)) {
        QPoint delta_pos = QCursor::pos() - m_mouse_start;
        double pixels_per_mspace = (m_window_width > m_window_height) ? double(m_window_width)*m_zoom : double(m_window_height)*m_zoom;
        double mspace_per_pixel = 2.0/pixels_per_mspace;
        double center_delta_x = double(delta_pos.x())*mspace_per_pixel;
        double center_delta_y = double(delta_pos.y())*mspace_per_pixel;
        m_center_x = m_center_start_x - center_delta_x;
        m_center_y = m_center_start_y + center_delta_y;
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
    if(event->button() == Qt::LeftButton) {
        m_mouse_start = QCursor::pos();
        m_center_start_x = m_center_x;
        m_center_start_y = m_center_y;
    }
    Input::registerMousePress(event->button());
}

void Window::mouseReleaseEvent(QMouseEvent *event)
{
    Input::registerMouseRelease(event->button());
}

void Window::wheelEvent(QWheelEvent *event)
{
    if(event->delta() > 0) {
        m_zoom *= 1.1;
    }
    else {
        m_zoom /= 1.1;
    }
    printf("zoom=%g\n",m_zoom);
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
