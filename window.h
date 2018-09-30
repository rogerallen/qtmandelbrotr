#ifndef WINDOW_H
#define WINDOW_H

#include <QOpenGLWindow>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>

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

private:
    // OpenGL State Information
    QOpenGLBuffer             m_vertex;
    QOpenGLVertexArrayObject  m_object;
    QOpenGLShaderProgram     *m_program;
    QOpenGLTexture           *m_texture;
    GLuint                    m_shared_pbo_id, m_shared_tex_id;
    int                       m_shared_width, m_shared_height;
    void                     *m_cuda_pbo_handle;
    float m_t;

    void printInfo();
};

#endif // WINDOW_H
