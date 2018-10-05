#ifndef WINDOW_H
#define WINDOW_H

#include <QOpenGLWindow>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>

#include "glm/mat4x4.hpp"

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

class Window : public QOpenGLWindow,
        protected QOpenGLFunctions
{
    Q_OBJECT

public:
    Window();
    ~Window();

    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void teardownGL();

protected slots:
    void update();

protected:
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);

private:
    // OpenGL State Information
    int                       u_cameraToView;
    glm::mat4                 m_projection;
    QOpenGLBuffer             m_vertex;
    QOpenGLVertexArrayObject  m_object;
    QOpenGLShaderProgram     *m_program;
    QOpenGLTexture           *m_texture;
    GLuint                    m_shared_pbo_id, m_shared_tex_id;
    int                       m_window_width, m_window_height;
    int                       m_shared_width, m_shared_height;
    void                     *m_cuda_pbo_handle;
    double m_zoom;
    bool   m_mouse_down;
    QPoint m_mouse_start;
    double m_center_start_x, m_center_start_y, m_center_x, m_center_y;
    bool   m_switch_fullscreen, m_is_full_screen;
    bool   m_zoom_out_mode;
    int    m_iter;
    bool   m_double_precision;
    bool   m_quit;
    unsigned char *m_pixels;
    bool   m_save_image;

    void printInfo();
};

#endif // WINDOW_H
