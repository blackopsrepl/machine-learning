# frozen_string_literal: true

# The parts of Box2D 2.3 that Gym's Lunar Lander exercises, ported to Ruby:
# polygon mass properties, edge-versus-polygon contact manifolds (b2EPCollider
# for edges without neighbours), the sequential-impulse contact solver with
# friction, the two-point block solver and Baumgarte position correction,
# revolute joints with a motor and angle limits, and island sleeping.
# Continuous collision (time-of-impact sub-stepping) is not ported: contacts
# are found and resolved by the discrete solver at each step.
module Box2D
  LINEAR_SLOP = 0.005
  ANGULAR_SLOP = 2.0 / 180 * Math::PI
  POLYGON_RADIUS = 2 * LINEAR_SLOP
  MAX_LINEAR_CORRECTION = 0.2
  MAX_ANGULAR_CORRECTION = 8.0 / 180 * Math::PI
  MAX_TRANSLATION = 2.0
  MAX_ROTATION = 0.5 * Math::PI
  BAUMGARTE = 0.2
  VELOCITY_THRESHOLD = 1.0
  TIME_TO_SLEEP = 0.5
  LINEAR_SLEEP_TOLERANCE = 0.01
  ANGULAR_SLEEP_TOLERANCE = 2.0 / 180 * Math::PI
  MAX_CONDITION_NUMBER = 1000.0

  module_function

  def cross(a, b) = a[0] * b[1] - a[1] * b[0]
  def cross_sv(s, v) = [-s * v[1], s * v[0]]
  def dot(a, b) = a[0] * b[0] + a[1] * b[1]
  def rotate(c, s, v) = [c * v[0] - s * v[1], s * v[0] + c * v[1]]
  def rotate_t(c, s, v) = [c * v[0] + s * v[1], -s * v[0] + c * v[1]]

  # Convex polygon in body coordinates: gift-wrapped hull starting at the
  # rightmost vertex, outward normals, centroid, radius 0.01.
  class Polygon
    attr_reader :vertices, :normals, :centroid

    def self.box(hx, hy) = new([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])

    def initialize(points)
      i0 = points.each_index.max_by { |i| [points[i][0], -points[i][1]] }
      hull = []
      ih = i0
      loop do
        hull << ih
        ie = 0
        (1...points.size).each do |j|
          if ie == ih
            ie = j
            next
          end
          r = [points[ie][0] - points[ih][0], points[ie][1] - points[ih][1]]
          v = [points[j][0] - points[ih][0], points[j][1] - points[ih][1]]
          c = Box2D.cross(r, v)
          ie = j if c.negative? || (c.zero? && Box2D.dot(v, v) > Box2D.dot(r, r))
        end
        ih = ie
        break if ie == i0
      end
      @vertices = hull.map { |i| points[i].map(&:to_f) }
      @normals = @vertices.each_index.map do |i|
        a, b = @vertices[i], @vertices[(i + 1) % @vertices.size]
        e = [b[0] - a[0], b[1] - a[1]]
        length = Math.hypot(*e)
        [e[1] / length, -e[0] / length]
      end
      @centroid = mass(1.0)[1]
    end

    def radius = POLYGON_RADIUS

    # Triangle fan from the first vertex: mass, centre, inertia about origin.
    def mass(density)
      s = @vertices.first
      area = 0.0
      center = [0.0, 0.0]
      inertia = 0.0
      @vertices.each_index do |i|
        e1 = [@vertices[i][0] - s[0], @vertices[i][1] - s[1]]
        b = @vertices[(i + 1) % @vertices.size]
        e2 = [b[0] - s[0], b[1] - s[1]]
        d = Box2D.cross(e1, e2)
        triangle = 0.5 * d
        area += triangle
        center[0] += triangle / 3.0 * (e1[0] + e2[0])
        center[1] += triangle / 3.0 * (e1[1] + e2[1])
        intx2 = e1[0] * e1[0] + e2[0] * e1[0] + e2[0] * e2[0]
        inty2 = e1[1] * e1[1] + e2[1] * e1[1] + e2[1] * e2[1]
        inertia += 0.25 / 3.0 * d * (intx2 + inty2)
      end
      center = [center[0] / area, center[1] / area]
      world_center = [center[0] + s[0], center[1] + s[1]]
      mass = density * area
      i = density * inertia + mass * (Box2D.dot(world_center, world_center) - Box2D.dot(center, center))
      [mass, world_center, i]
    end
  end

  Edge = Struct.new(:v1, :v2) do
    def radius = POLYGON_RADIUS
  end

  class Fixture
    attr_reader :shape, :body, :friction, :restitution, :density, :category, :mask

    def initialize(body, shape, density: 0.0, friction: 0.2, restitution: 0.0, category: 0x0001, mask: 0xffff)
      @body, @shape, @density, @friction, @restitution, @category, @mask = body, shape, density, friction, restitution, category, mask
    end

    def collides_with?(other) = (@mask & other.category) != 0 && (other.mask & @category) != 0
  end

  class Body
    attr_accessor :position, :angle, :linear_velocity, :angular_velocity, :awake, :sleep_time, :user_data
    attr_reader :type, :fixtures, :inv_mass, :inv_i, :local_center, :center, :force, :torque

    def initialize(type, position, angle)
      @type, @position, @angle = type, position.map(&:to_f), angle.to_f
      @linear_velocity, @angular_velocity = [0.0, 0.0], 0.0
      @force, @torque = [0.0, 0.0], 0.0
      @fixtures = []
      @awake, @sleep_time = true, 0.0
      @inv_mass = @inv_i = 0.0
      @local_center = [0.0, 0.0]
      @center = @position.dup
    end

    def dynamic? = @type == :dynamic

    def add_fixture(shape, **options)
      fixture = Fixture.new(self, shape, **options)
      @fixtures << fixture
      reset_mass if dynamic?
      fixture
    end

    def reset_mass
      mass = 0.0
      center = [0.0, 0.0]
      inertia = 0.0
      @fixtures.each do |fixture|
        next unless fixture.density.positive?

        m, c, i = fixture.shape.mass(fixture.density)
        mass += m
        center = [center[0] + m * c[0], center[1] + m * c[1]]
        inertia += i
      end
      @local_center = [center[0] / mass, center[1] / mass]
      inertia -= mass * Box2D.dot(@local_center, @local_center)
      @inv_mass = 1.0 / mass
      @inv_i = 1.0 / inertia
      @center = world_point(@local_center)
    end

    def cos = Math.cos(@angle)
    def sin = Math.sin(@angle)

    def world_point(local)
      r = Box2D.rotate(cos, sin, local)
      [@position[0] + r[0], @position[1] + r[1]]
    end

    def set_awake(flag)
      if flag
        @sleep_time = 0.0 unless @awake
        @awake = true
      else
        @awake = false
        @sleep_time = 0.0
        @linear_velocity, @angular_velocity = [0.0, 0.0], 0.0
        @force, @torque = [0.0, 0.0], 0.0
      end
    end

    def apply_force_to_center(force, wake: true)
      set_awake(true) if wake && !@awake
      @force = [@force[0] + force[0], @force[1] + force[1]] if @awake
    end

    def apply_linear_impulse(impulse, point, wake: true)
      set_awake(true) if wake && !@awake
      return unless @awake

      @linear_velocity = [@linear_velocity[0] + @inv_mass * impulse[0], @linear_velocity[1] + @inv_mass * impulse[1]]
      @angular_velocity += @inv_i * Box2D.cross([point[0] - @center[0], point[1] - @center[1]], impulse)
    end

    def clear_forces
      @force = [0.0, 0.0]
      @torque = 0.0
    end

    # Keep the origin consistent with the solved centre of mass and angle.
    def synchronize(center, angle)
      @center, @angle = center, angle
      r = Box2D.rotate(Math.cos(angle), Math.sin(angle), @local_center)
      @position = [center[0] - r[0], center[1] - r[1]]
    end
  end

  # Revolute joint: point constraint, motor with maximum torque, angle limits.
  class RevoluteJoint
    attr_reader :body_a, :body_b

    def initialize(body_a, body_b, anchor_a:, anchor_b:, lower:, upper:, motor_speed:, max_motor_torque:, reference_angle: 0.0)
      @body_a, @body_b = body_a, body_b
      @anchor_a, @anchor_b = anchor_a.map(&:to_f), anchor_b.map(&:to_f)
      @lower, @upper, @motor_speed, @max_torque, @reference = lower, upper, motor_speed, max_motor_torque, reference_angle
      @impulse = [0.0, 0.0, 0.0]
      @motor_impulse = 0.0
      @state = :inactive
    end

    def init_velocity(pos, vel, dt)
      a, b = @body_a, @body_b
      aa, ab = pos[a][1], pos[b][1]
      @ra = Box2D.rotate(Math.cos(aa), Math.sin(aa), [@anchor_a[0] - a.local_center[0], @anchor_a[1] - a.local_center[1]])
      @rb = Box2D.rotate(Math.cos(ab), Math.sin(ab), [@anchor_b[0] - b.local_center[0], @anchor_b[1] - b.local_center[1]])
      ma, mb, ia, ib = a.inv_mass, b.inv_mass, a.inv_i, b.inv_i
      ra, rb = @ra, @rb
      @k = [
        [ma + mb + ra[1] * ra[1] * ia + rb[1] * rb[1] * ib, -ra[1] * ra[0] * ia - rb[1] * rb[0] * ib, -ra[1] * ia - rb[1] * ib],
        [-ra[1] * ra[0] * ia - rb[1] * rb[0] * ib, ma + mb + ra[0] * ra[0] * ia + rb[0] * rb[0] * ib, ra[0] * ia + rb[0] * ib],
        [-ra[1] * ia - rb[1] * ib, ra[0] * ia + rb[0] * ib, ia + ib]
      ]
      @motor_mass = ia + ib
      @motor_mass = 1.0 / @motor_mass if @motor_mass.positive?
      angle = ab - aa - @reference
      if (@upper - @lower).abs < 2 * ANGULAR_SLOP
        @state = :equal
      elsif angle <= @lower
        @impulse[2] = 0.0 unless @state == :lower
        @state = :lower
      elsif angle >= @upper
        @impulse[2] = 0.0 unless @state == :upper
        @state = :upper
      else
        @state = :inactive
        @impulse[2] = 0.0
      end
      p = [@impulse[0], @impulse[1]]
      va, wa = vel[a]
      vb, wb = vel[b]
      vel[a] = [[va[0] - ma * p[0], va[1] - ma * p[1]], wa - ia * (Box2D.cross(ra, p) + @motor_impulse + @impulse[2])]
      vel[b] = [[vb[0] + mb * p[0], vb[1] + mb * p[1]], wb + ib * (Box2D.cross(rb, p) + @motor_impulse + @impulse[2])]
      @dt = dt
    end

    def solve22(b)
      a11, a12, a21, a22 = @k[0][0], @k[0][1], @k[1][0], @k[1][1]
      det = a11 * a22 - a12 * a21
      det = 1.0 / det unless det.zero?
      [det * (a22 * b[0] - a12 * b[1]), det * (a11 * b[1] - a21 * b[0])]
    end

    def solve33(b)
      ex = [@k[0][0], @k[1][0], @k[2][0]]
      ey = [@k[0][1], @k[1][1], @k[2][1]]
      ez = [@k[0][2], @k[1][2], @k[2][2]]
      c3 = ->(u, v) { [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]] }
      d3 = ->(u, v) { u[0] * v[0] + u[1] * v[1] + u[2] * v[2] }
      det = d3.call(ex, c3.call(ey, ez))
      det = 1.0 / det unless det.zero?
      [det * d3.call(b, c3.call(ey, ez)), det * d3.call(ex, c3.call(b, ez)), det * d3.call(ex, c3.call(ey, b))]
    end

    def solve_velocity(vel)
      a, b = @body_a, @body_b
      ma, mb, ia, ib = a.inv_mass, b.inv_mass, a.inv_i, b.inv_i
      va, wa = vel[a]
      vb, wb = vel[b]
      if @state != :equal
        impulse = -@motor_mass * (wb - wa - @motor_speed)
        old = @motor_impulse
        limit = @dt * @max_torque
        @motor_impulse = (@motor_impulse + impulse).clamp(-limit, limit)
        impulse = @motor_impulse - old
        wa -= ia * impulse
        wb += ib * impulse
      end
      cdot1 = [vb[0] - wb * @rb[1] - va[0] + wa * @ra[1], vb[1] + wb * @rb[0] - va[1] - wa * @ra[0]]
      if @state != :inactive
        impulse = solve33([cdot1[0], cdot1[1], wb - wa]).map(&:-@)
        if @state == :equal
          @impulse = @impulse.zip(impulse).map(&:sum)
        else
          new_impulse = @impulse[2] + impulse[2]
          if (@state == :lower && new_impulse.negative?) || (@state == :upper && new_impulse.positive?)
            reduced = solve22([-cdot1[0] + @impulse[2] * @k[0][2], -cdot1[1] + @impulse[2] * @k[1][2]])
            impulse = [reduced[0], reduced[1], -@impulse[2]]
            @impulse = [@impulse[0] + reduced[0], @impulse[1] + reduced[1], 0.0]
          else
            @impulse = @impulse.zip(impulse).map(&:sum)
          end
        end
        p = [impulse[0], impulse[1]]
        va = [va[0] - ma * p[0], va[1] - ma * p[1]]
        wa -= ia * (Box2D.cross(@ra, p) + impulse[2])
        vb = [vb[0] + mb * p[0], vb[1] + mb * p[1]]
        wb += ib * (Box2D.cross(@rb, p) + impulse[2])
      else
        impulse = solve22([-cdot1[0], -cdot1[1]])
        @impulse[0] += impulse[0]
        @impulse[1] += impulse[1]
        va = [va[0] - ma * impulse[0], va[1] - ma * impulse[1]]
        wa -= ia * Box2D.cross(@ra, impulse)
        vb = [vb[0] + mb * impulse[0], vb[1] + mb * impulse[1]]
        wb += ib * Box2D.cross(@rb, impulse)
      end
      vel[a] = [va, wa]
      vel[b] = [vb, wb]
    end

    def solve_position(pos)
      a, b = @body_a, @body_b
      ca, aa = pos[a]
      cb, ab = pos[b]
      angular_error = 0.0
      if @state != :inactive
        angle = ab - aa - @reference
        limit_impulse = 0.0
        case @state
        when :equal
          c = (angle - @lower).clamp(-MAX_ANGULAR_CORRECTION, MAX_ANGULAR_CORRECTION)
          limit_impulse = -@motor_mass * c
          angular_error = c.abs
        when :lower
          c = angle - @lower
          angular_error = -c
          limit_impulse = -@motor_mass * (c + ANGULAR_SLOP).clamp(-MAX_ANGULAR_CORRECTION, 0.0)
        when :upper
          c = angle - @upper
          angular_error = c
          limit_impulse = -@motor_mass * (c - ANGULAR_SLOP).clamp(0.0, MAX_ANGULAR_CORRECTION)
        end
        aa -= a.inv_i * limit_impulse
        ab += b.inv_i * limit_impulse
      end
      ra = Box2D.rotate(Math.cos(aa), Math.sin(aa), [@anchor_a[0] - a.local_center[0], @anchor_a[1] - a.local_center[1]])
      rb = Box2D.rotate(Math.cos(ab), Math.sin(ab), [@anchor_b[0] - b.local_center[0], @anchor_b[1] - b.local_center[1]])
      c = [cb[0] + rb[0] - ca[0] - ra[0], cb[1] + rb[1] - ca[1] - ra[1]]
      position_error = Math.hypot(*c)
      ma, mb, ia, ib = a.inv_mass, b.inv_mass, a.inv_i, b.inv_i
      k11 = ma + mb + ia * ra[1] * ra[1] + ib * rb[1] * rb[1]
      k12 = -ia * ra[0] * ra[1] - ib * rb[0] * rb[1]
      k22 = ma + mb + ia * ra[0] * ra[0] + ib * rb[0] * rb[0]
      det = k11 * k22 - k12 * k12
      det = 1.0 / det unless det.zero?
      impulse = [-det * (k22 * c[0] - k12 * c[1]), -det * (k11 * c[1] - k12 * c[0])]
      pos[a] = [[ca[0] - ma * impulse[0], ca[1] - ma * impulse[1]], aa - ia * Box2D.cross(ra, impulse)]
      pos[b] = [[cb[0] + mb * impulse[0], cb[1] + mb * impulse[1]], ab + ib * Box2D.cross(rb, impulse)]
      position_error <= LINEAR_SLOP && angular_error <= ANGULAR_SLOP
    end
  end

  # Manifold point: local point, feature id, accumulated impulses.
  ManifoldPoint = Struct.new(:local_point, :id, :normal_impulse, :tangent_impulse)
  Manifold = Struct.new(:type, :local_normal, :local_point, :points)

  # b2CollideEdgeAndPolygon for an edge without ghost vertices. The polygon is
  # expressed in the edge body's frame; returns a Manifold (maybe pointless).
  def collide_edge_polygon(edge, xf_a, polygon, xf_b)
    ca, sa, pa = xf_a
    cb, sb, pb = xf_b
    # Relative transform of B in A: rotation angle_b-angle_a, offset.
    c = ca * cb + sa * sb
    s = ca * sb - sa * cb
    offset = rotate_t(ca, sa, [pb[0] - pa[0], pb[1] - pa[1]])
    to_a = ->(v) { r = rotate(c, s, v); [r[0] + offset[0], r[1] + offset[1]] }
    centroid = to_a.call(polygon.centroid)
    v1, v2 = edge.v1, edge.v2
    e = [v2[0] - v1[0], v2[1] - v1[1]]
    length = Math.hypot(*e)
    normal1 = [e[1] / length, -e[0] / length]
    front = dot(normal1, [centroid[0] - v1[0], centroid[1] - v1[1]]) >= 0
    normal = front ? normal1 : normal1.map(&:-@)
    limit = front ? normal1.map(&:-@) : normal1
    vertices = polygon.vertices.map { |v| to_a.call(v) }
    normals = polygon.normals.map { |n| rotate(c, s, n) }
    radius = 2 * POLYGON_RADIUS
    empty = Manifold.new(nil, nil, nil, [])
    edge_separation = vertices.map { |v| dot(normal, [v[0] - v1[0], v[1] - v1[1]]) }.min
    return empty if edge_separation > radius

    polygon_axis = nil
    perp = [-normal[1], normal[0]]
    normals.each_index do |i|
      n = normals[i].map(&:-@)
      s1 = dot(n, [vertices[i][0] - v1[0], vertices[i][1] - v1[1]])
      s2 = dot(n, [vertices[i][0] - v2[0], vertices[i][1] - v2[1]])
      separation = [s1, s2].min
      return empty if separation > radius

      next if dot([n[0] - limit[0], n[1] - limit[1]], normal) < -ANGULAR_SLOP && dot(n, perp) >= 0
      next if dot([n[0] - limit[0], n[1] - limit[1]], normal) < -ANGULAR_SLOP && dot(n, perp).negative?

      polygon_axis = [i, separation] if polygon_axis.nil? || separation > polygon_axis[1]
    end
    edge_reference = polygon_axis.nil? || !(polygon_axis[1] > 0.98 * edge_separation + 0.001)
    if edge_reference
      best = normals.each_index.min_by { |i| dot(normal, normals[i]) }
      i2 = (best + 1) % vertices.size
      incident = [[vertices[best], [0, best, :face, :vertex]], [vertices[i2], [0, i2, :face, :vertex]]]
      ref_i1, ref_i2 = front ? [0, 1] : [1, 0]
      ref_v1, ref_v2 = front ? [v1, v2] : [v2, v1]
      ref_normal = front ? normal1 : normal1.map(&:-@)
    else
      index = polygon_axis[0]
      incident = [[v1, [0, index, :vertex, :face]], [v2, [0, index, :vertex, :face]]]
      ref_i1 = index
      ref_i2 = (index + 1) % vertices.size
      ref_v1, ref_v2 = vertices[ref_i1], vertices[ref_i2]
      ref_normal = normals[ref_i1]
    end
    side1 = [ref_normal[1], -ref_normal[0]]
    side2 = side1.map(&:-@)
    clip = lambda do |points, side, offset, index|
      d0 = dot(side, points[0][0]) - offset
      d1 = dot(side, points[1][0]) - offset
      out = []
      out << points[0] if d0 <= 0
      out << points[1] if d1 <= 0
      if d0 * d1 < 0
        t = d0 / (d0 - d1)
        a, b = points[0][0], points[1][0]
        out << [[a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], [index, points[0][1][1], :vertex, :face]]
      end
      out
    end
    clipped = clip.call(incident, side1, dot(side1, ref_v1), ref_i1)
    return empty if clipped.size < 2

    clipped = clip.call(clipped, side2, dot(side2, ref_v2), ref_i2)
    return empty if clipped.size < 2

    manifold = if edge_reference
                 Manifold.new(:face_a, ref_normal, ref_v1, [])
               else
                 Manifold.new(:face_b, polygon.normals[ref_i1], polygon.vertices[ref_i1], [])
               end
    clipped.first(2).each do |point, id|
      next unless dot(ref_normal, [point[0] - ref_v1[0], point[1] - ref_v1[1]]) <= radius

      if edge_reference
        local = rotate_t(c, s, [point[0] - offset[0], point[1] - offset[1]])
        manifold.points << ManifoldPoint.new(local, id, 0.0, 0.0)
      else
        manifold.points << ManifoldPoint.new(point, [id[1], id[0], id[3], id[2]], 0.0, 0.0)
      end
    end
    manifold
  end

  Contact = Struct.new(:fixture_a, :fixture_b, :manifold, :touching)

  class World
    attr_reader :bodies, :joints
    attr_accessor :listener

    def initialize(gravity: [0.0, -10.0])
      @gravity = gravity
      @bodies, @joints = [], []
      @contacts = {}
    end

    def create_body(type, position: [0.0, 0.0], angle: 0.0)
      Body.new(type, position, angle).tap { |b| @bodies << b }
    end

    def create_joint(joint) = joint.tap { @joints << joint }

    def transform(body)
      [Math.cos(body.angle), Math.sin(body.angle), body.position]
    end

    # b2ContactManager::Collide: update every edge/polygon pair, keep
    # impulses of points whose feature ids persist, report touch changes.
    def collide
      @bodies.each do |a|
        next if a.dynamic?

        a.fixtures.each do |fa|
          next unless fa.shape.is_a?(Edge)

          @bodies.each do |b|
            next unless b.dynamic? && b.awake

            b.fixtures.each do |fb|
              next unless fb.shape.is_a?(Polygon) && fa.collides_with?(fb)

              key = [fa.object_id, fb.object_id]
              contact = @contacts[key] ||= Contact.new(fa, fb, Manifold.new(nil, nil, nil, []), false)
              old = contact.manifold
              manifold = Box2D.collide_edge_polygon(fa.shape, transform(a), fb.shape, transform(b))
              manifold.points.each do |point|
                match = old.points.find { |o| o.id == point.id }
                next unless match

                point.normal_impulse = match.normal_impulse
                point.tangent_impulse = match.tangent_impulse
              end
              touching = !manifold.points.empty?
              was = contact.touching
              contact.manifold, contact.touching = manifold, touching
              @listener&.begin_contact(contact) if touching && !was
              @listener&.end_contact(contact) if was && !touching
            end
          end
        end
      end
    end

    def step(dt, velocity_iterations, position_iterations)
      collide
      solve(dt, velocity_iterations, position_iterations)
      @bodies.each(&:clear_forces)
    end

    private

    def solve(dt, velocity_iterations, position_iterations)
      island = @bodies.select { |b| b.dynamic? && b.awake }
      return if island.empty?

      contacts = @contacts.values.select { |c| c.touching && c.fixture_b.body.awake }
      pos, vel = {}, {}
      island.each do |b|
        v = b.linear_velocity
        w = b.angular_velocity
        v = [v[0] + dt * (@gravity[0] + b.inv_mass * b.force[0]), v[1] + dt * (@gravity[1] + b.inv_mass * b.force[1])]
        w += dt * b.inv_i * b.torque
        pos[b] = [b.center.dup, b.angle]
        vel[b] = [v, w]
      end
      @bodies.reject(&:dynamic?).each do |b|
        pos[b] = [b.center.dup, b.angle]
        vel[b] = [[0.0, 0.0], 0.0]
      end
      constraints = contacts.map { |contact| contact_constraint(contact, pos, vel) }
      constraints.each { |cc| warm_start(cc, vel) }
      joints = @joints.select { |j| j.body_a.awake || j.body_b.awake }
      joints.each { |j| j.init_velocity(pos, vel, dt) }
      velocity_iterations.times do
        joints.each { |j| j.solve_velocity(vel) }
        constraints.each { |cc| solve_contact_velocity(cc, vel) }
      end
      constraints.each { |cc| cc[:points].each { |p| p[:source].normal_impulse, p[:source].tangent_impulse = p[:normal_impulse], p[:tangent_impulse] } }
      island.each do |b|
        v, w = vel[b]
        translation = [dt * v[0], dt * v[1]]
        if Box2D.dot(translation, translation) > MAX_TRANSLATION**2
          ratio = MAX_TRANSLATION / Math.hypot(*translation)
          v = [v[0] * ratio, v[1] * ratio]
        end
        w *= MAX_ROTATION / (dt * w).abs if (dt * w)**2 > MAX_ROTATION**2
        c, a = pos[b]
        pos[b] = [[c[0] + dt * v[0], c[1] + dt * v[1]], a + dt * w]
        vel[b] = [v, w]
      end
      solved = false
      position_iterations.times do
        contacts_ok = solve_contact_positions(constraints, pos)
        joints_ok = joints.map { |j| j.solve_position(pos) }.all?
        if contacts_ok && joints_ok
          solved = true
          break
        end
      end
      island.each do |b|
        c, a = pos[b]
        b.synchronize(c, a)
        b.linear_velocity, b.angular_velocity = vel[b]
      end
      sleep_island(island, dt, solved)
    end

    def sleep_island(island, dt, solved)
      min_sleep = Float::INFINITY
      island.each do |b|
        v = b.linear_velocity
        if b.angular_velocity**2 > ANGULAR_SLEEP_TOLERANCE**2 || Box2D.dot(v, v) > LINEAR_SLEEP_TOLERANCE**2
          b.sleep_time = 0.0
          min_sleep = 0.0
        else
          b.sleep_time += dt
          min_sleep = [min_sleep, b.sleep_time].min
        end
      end
      island.each { |b| b.set_awake(false) } if min_sleep >= TIME_TO_SLEEP && solved
    end

    def world_manifold(manifold, xf_a, ra, xf_b, rb)
      ca, sa, pa = xf_a
      cb, sb, pb = xf_b
      point_in = ->(c, s, p, local) { r = Box2D.rotate(c, s, local); [p[0] + r[0], p[1] + r[1]] }
      if manifold.type == :face_a
        normal = Box2D.rotate(ca, sa, manifold.local_normal)
        plane = point_in.call(ca, sa, pa, manifold.local_point)
        points = manifold.points.map do |mp|
          clip = point_in.call(cb, sb, pb, mp.local_point)
          d = ra - Box2D.dot([clip[0] - plane[0], clip[1] - plane[1]], normal)
          point_a = [clip[0] + d * normal[0], clip[1] + d * normal[1]]
          point_b = [clip[0] - rb * normal[0], clip[1] - rb * normal[1]]
          [0.5 * (point_a[0] + point_b[0]), 0.5 * (point_a[1] + point_b[1])]
        end
        [normal, points]
      else
        normal = Box2D.rotate(cb, sb, manifold.local_normal)
        plane = point_in.call(cb, sb, pb, manifold.local_point)
        points = manifold.points.map do |mp|
          clip = point_in.call(ca, sa, pa, mp.local_point)
          d = rb - Box2D.dot([clip[0] - plane[0], clip[1] - plane[1]], normal)
          point_b = [clip[0] + d * normal[0], clip[1] + d * normal[1]]
          point_a = [clip[0] - ra * normal[0], clip[1] - ra * normal[1]]
          [0.5 * (point_a[0] + point_b[0]), 0.5 * (point_a[1] + point_b[1])]
        end
        [normal.map(&:-@), points]
      end
    end

    def xf_from(body, pos)
      c, a = pos[body]
      cs, sn = Math.cos(a), Math.sin(a)
      r = Box2D.rotate(cs, sn, body.local_center)
      [cs, sn, [c[0] - r[0], c[1] - r[1]]]
    end

    def contact_constraint(contact, pos, vel)
      a, b = contact.fixture_a.body, contact.fixture_b.body
      normal, points = world_manifold(contact.manifold, xf_from(a, pos), contact.fixture_a.shape.radius, xf_from(b, pos), contact.fixture_b.shape.radius)
      ma, mb, ia, ib = a.inv_mass, b.inv_mass, a.inv_i, b.inv_i
      tangent = [normal[1], -normal[0]]
      ca, cb = pos[a][0], pos[b][0]
      va, wa = vel[a]
      vb, wb = vel[b]
      restitution = [contact.fixture_a.restitution, contact.fixture_b.restitution].max
      cps = contact.manifold.points.each_with_index.map do |mp, i|
        r_a = [points[i][0] - ca[0], points[i][1] - ca[1]]
        r_b = [points[i][0] - cb[0], points[i][1] - cb[1]]
        rna, rnb = Box2D.cross(r_a, normal), Box2D.cross(r_b, normal)
        kn = ma + mb + ia * rna * rna + ib * rnb * rnb
        rta, rtb = Box2D.cross(r_a, tangent), Box2D.cross(r_b, tangent)
        kt = ma + mb + ia * rta * rta + ib * rtb * rtb
        dv = [vb[0] - wb * r_b[1] - va[0] + wa * r_a[1], vb[1] + wb * r_b[0] - va[1] - wa * r_a[0]]
        relative = Box2D.dot(normal, dv)
        { source: mp, r_a: r_a, r_b: r_b, normal_impulse: mp.normal_impulse, tangent_impulse: mp.tangent_impulse,
          normal_mass: kn.positive? ? 1.0 / kn : 0.0, tangent_mass: kt.positive? ? 1.0 / kt : 0.0,
          bias: relative < -VELOCITY_THRESHOLD ? -restitution * relative : 0.0 }
      end
      cc = { a: a, b: b, normal: normal, tangent: tangent, friction: Math.sqrt(contact.fixture_a.friction * contact.fixture_b.friction), points: cps, manifold: contact.manifold, radius: contact.fixture_a.shape.radius + contact.fixture_b.shape.radius }
      if cps.size == 2
        p1, p2 = cps
        rn1a, rn1b = Box2D.cross(p1[:r_a], normal), Box2D.cross(p1[:r_b], normal)
        rn2a, rn2b = Box2D.cross(p2[:r_a], normal), Box2D.cross(p2[:r_b], normal)
        k11 = ma + mb + ia * rn1a * rn1a + ib * rn1b * rn1b
        k22 = ma + mb + ia * rn2a * rn2a + ib * rn2b * rn2b
        k12 = ma + mb + ia * rn1a * rn2a + ib * rn1b * rn2b
        if k11 * k11 < MAX_CONDITION_NUMBER * (k11 * k22 - k12 * k12)
          det = k11 * k22 - k12 * k12
          det = 1.0 / det unless det.zero?
          cc[:k] = [[k11, k12], [k12, k22]]
          cc[:normal_mass] = [[det * k22, -det * k12], [-det * k12, det * k11]]
        else
          cc[:points] = [p1]
        end
      end
      cc
    end

    def apply(cc, vel, impulse, r_a, r_b)
      a, b = cc[:a], cc[:b]
      va, wa = vel[a]
      vb, wb = vel[b]
      vel[a] = [[va[0] - a.inv_mass * impulse[0], va[1] - a.inv_mass * impulse[1]], wa - a.inv_i * Box2D.cross(r_a, impulse)]
      vel[b] = [[vb[0] + b.inv_mass * impulse[0], vb[1] + b.inv_mass * impulse[1]], wb + b.inv_i * Box2D.cross(r_b, impulse)]
    end

    def warm_start(cc, vel)
      n, t = cc[:normal], cc[:tangent]
      cc[:points].each do |p|
        impulse = [p[:normal_impulse] * n[0] + p[:tangent_impulse] * t[0], p[:normal_impulse] * n[1] + p[:tangent_impulse] * t[1]]
        apply(cc, vel, impulse, p[:r_a], p[:r_b])
      end
    end

    def relative_velocity(cc, vel, p)
      va, wa = vel[cc[:a]]
      vb, wb = vel[cc[:b]]
      [vb[0] - wb * p[:r_b][1] - va[0] + wa * p[:r_a][1], vb[1] + wb * p[:r_b][0] - va[1] - wa * p[:r_a][0]]
    end

    def solve_contact_velocity(cc, vel)
      n, t = cc[:normal], cc[:tangent]
      cc[:points].each do |p|
        vt = Box2D.dot(relative_velocity(cc, vel, p), t)
        lambda = -p[:tangent_mass] * vt
        max_friction = cc[:friction] * p[:normal_impulse]
        new_impulse = (p[:tangent_impulse] + lambda).clamp(-max_friction, max_friction)
        lambda = new_impulse - p[:tangent_impulse]
        p[:tangent_impulse] = new_impulse
        apply(cc, vel, [lambda * t[0], lambda * t[1]], p[:r_a], p[:r_b])
      end
      if cc[:points].size == 1
        p = cc[:points].first
        vn = Box2D.dot(relative_velocity(cc, vel, p), n)
        lambda = -p[:normal_mass] * (vn - p[:bias])
        new_impulse = [p[:normal_impulse] + lambda, 0.0].max
        lambda = new_impulse - p[:normal_impulse]
        p[:normal_impulse] = new_impulse
        apply(cc, vel, [lambda * n[0], lambda * n[1]], p[:r_a], p[:r_b])
        return
      end
      p1, p2 = cc[:points]
      k, m = cc[:k], cc[:normal_mass]
      a = [p1[:normal_impulse], p2[:normal_impulse]]
      vn1 = Box2D.dot(relative_velocity(cc, vel, p1), n)
      vn2 = Box2D.dot(relative_velocity(cc, vel, p2), n)
      b = [vn1 - p1[:bias] - (k[0][0] * a[0] + k[0][1] * a[1]), vn2 - p2[:bias] - (k[1][0] * a[0] + k[1][1] * a[1])]
      finish = lambda do |x|
        d = [x[0] - a[0], x[1] - a[1]]
        apply(cc, vel, [d[0] * n[0], d[0] * n[1]], p1[:r_a], p1[:r_b])
        apply(cc, vel, [d[1] * n[0], d[1] * n[1]], p2[:r_a], p2[:r_b])
        p1[:normal_impulse], p2[:normal_impulse] = x
      end
      x = [-(m[0][0] * b[0] + m[0][1] * b[1]), -(m[1][0] * b[0] + m[1][1] * b[1])]
      return finish.call(x) if x[0] >= 0 && x[1] >= 0

      x = [-p1[:normal_mass] * b[0], 0.0]
      return finish.call(x) if x[0] >= 0 && k[1][0] * x[0] + b[1] >= 0

      x = [0.0, -p2[:normal_mass] * b[1]]
      return finish.call(x) if x[1] >= 0 && k[0][1] * x[1] + b[0] >= 0

      finish.call([0.0, 0.0]) if b[0] >= 0 && b[1] >= 0
    end

    def solve_contact_positions(constraints, pos)
      min_separation = 0.0
      constraints.each do |cc|
        a, b = cc[:a], cc[:b]
        ma, mb, ia, ib = a.inv_mass, b.inv_mass, a.inv_i, b.inv_i
        manifold = cc[:manifold]
        cc[:points].each do |p|
          xa, xb = xf_from(a, pos), xf_from(b, pos)
          if manifold.type == :face_a
            normal = Box2D.rotate(xa[0], xa[1], manifold.local_normal)
            plane = Box2D.rotate(xa[0], xa[1], manifold.local_point).then { |r| [xa[2][0] + r[0], xa[2][1] + r[1]] }
            clip = Box2D.rotate(xb[0], xb[1], p[:source].local_point).then { |r| [xb[2][0] + r[0], xb[2][1] + r[1]] }
          else
            normal = Box2D.rotate(xb[0], xb[1], manifold.local_normal)
            plane = Box2D.rotate(xb[0], xb[1], manifold.local_point).then { |r| [xb[2][0] + r[0], xb[2][1] + r[1]] }
            clip = Box2D.rotate(xa[0], xa[1], p[:source].local_point).then { |r| [xa[2][0] + r[0], xa[2][1] + r[1]] }
          end
          separation = Box2D.dot([clip[0] - plane[0], clip[1] - plane[1]], normal) - cc[:radius]
          normal = normal.map(&:-@) if manifold.type == :face_b
          ca, aa = pos[a]
          cb, ab = pos[b]
          r_a = [clip[0] - ca[0], clip[1] - ca[1]]
          r_b = [clip[0] - cb[0], clip[1] - cb[1]]
          min_separation = [min_separation, separation].min
          c = (BAUMGARTE * (separation + LINEAR_SLOP)).clamp(-MAX_LINEAR_CORRECTION, 0.0)
          rna, rnb = Box2D.cross(r_a, normal), Box2D.cross(r_b, normal)
          k = ma + mb + ia * rna * rna + ib * rnb * rnb
          impulse = k.positive? ? -c / k : 0.0
          pp = [impulse * normal[0], impulse * normal[1]]
          pos[a] = [[ca[0] - ma * pp[0], ca[1] - ma * pp[1]], aa - ia * Box2D.cross(r_a, pp)]
          pos[b] = [[cb[0] + mb * pp[0], cb[1] + mb * pp[1]], ab + ib * Box2D.cross(r_b, pp)]
        end
      end
      min_separation >= -3.0 * LINEAR_SLOP
    end
  end
end
